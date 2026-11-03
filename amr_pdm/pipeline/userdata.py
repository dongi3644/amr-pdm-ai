"""사용자 데이터: CSV → 파이프라인이 먹는 1초 창 → 학습·진단

    data = load("user_data")                       # 파일 또는 폴더. 형식 설명은 DATA_FORMAT.md
    table, monitor = diagnose(pipe, data.robots["R1"])    # 회차별 진단 결과
    pipe, notes = train_user(data)                 # 내 데이터로 다시 학습

한 줄이 진동 샘플 하나인 CSV를 읽어 시뮬레이터(sim/simulator.py)와 같은 창 형식으로 바꾼다.
그 뒤의 특징 추출·모델은 시뮬레이션 때와 같은 코드를 쓴다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from pipeline.features import extract
from pipeline.models import label_ko
from pipeline.system import AMRPipeline

EXTENSIONS = (".csv", ".csv.gz", ".zip")
SAMPLE_PREFIX = "sample_"
KIND_NAMES = {"healthy": "normal", "ok": "normal", "none": "normal", "정상": "normal",
              **{v: k for k, v in C.KIND_KO.items()}}
WHEEL_NAMES = {"left": "left", "l": "left", "좌": "left", "좌측": "left", "왼쪽": "left",
               "right": "right", "r": "right", "우": "right", "우측": "right", "오른쪽": "right"}


class UserDataError(ValueError):
    """사용자가 고칠 수 있는 데이터 문제. 메시지를 그대로 보여 준다."""


class UserData:
    def __init__(self):
        self.robots, self.warnings, self.fs = {}, [], {}   # robots: 로봇 이름 → 1초 창 목록

    def warn(self, text):
        if text not in self.warnings:
            self.warnings.append(text)

    def summary(self):
        rows = []
        for name, recs in self.robots.items():
            kinds = sorted({r["true_kind"] for r in recs if r["labeled"] and r["true_kind"] != "normal"})
            labeled = any(r["labeled"] for r in recs)
            rows.append({"로봇": name, "회차": recs[-1]["lap"] + 1, "1초 창": len(recs), "진동 샘플링(Hz)": round(self.fs[name], 1),
                         "구간": len({r["segment"] for r in recs}),
                         "라벨": ", ".join(C.KIND_KO.get(k, k) for k in kinds) if kinds else ("정상" if labeled else "없음")})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- 파일 읽기
def list_files(path=C.USER_DATA_DIR):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise UserDataError(f"경로를 찾을 수 없습니다: {path}")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith(EXTENSIONS))
    if not files:
        raise UserDataError(f"{path} 폴더에 CSV 파일이 없습니다. 형식은 DATA_FORMAT.md를 보세요.")
    return files


def read_table(path):
    for enc in ("utf-8-sig", "cp949"):                     # 엑셀에서 저장한 한글 CSV는 cp949일 수 있음
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError) as e:
            raise UserDataError(f"{Path(path).name}: CSV로 읽지 못했습니다 ({e})") from e
    else:
        raise UserDataError(f"{Path(path).name}: 글자 인코딩을 알 수 없습니다. UTF-8로 저장해 주세요.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for name, aliases in C.COLUMN_ALIASES.items():
        if name not in df.columns:
            found = next((a for a in aliases if a in df.columns), None)
            if found:
                rename[found] = name
    return df.rename(columns=rename)


def _numeric(df, name, data, fname):
    """숫자 열을 배열로. 열이 없거나 전부 비어 있으면 None"""
    if name not in df.columns:
        return None
    col = pd.to_numeric(df[name], errors="coerce")
    bad = int((col.isna() & df[name].notna()).sum())
    if bad:
        data.warn(f"{fname}: {name} 열에 숫자가 아닌 값 {bad}개를 빈칸으로 처리했습니다.")
    return col.to_numpy(float) if col.notna().any() else None


def _seconds(col, fname):
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.99:
        return num.to_numpy(float)
    dt = pd.to_datetime(col, errors="coerce")
    if dt.notna().mean() > 0.99:
        return (dt - dt.iloc[0]).dt.total_seconds().to_numpy(float)
    raise UserDataError(f"{fname}: time 열을 시간으로 읽지 못했습니다. 초 단위 숫자나 날짜·시각 형식이어야 합니다.")


def _mean(values, prev):
    v = values[~np.isnan(values)]
    return float(v.mean()) if len(v) else prev


def _labels(col, names):
    return col.fillna("").astype(str).str.strip().str.lower().map(lambda s: names.get(s, s)).to_numpy()


def _windows(df, fname, data, fs_arg, lap_s):
    """로봇 1대의 표 → 1초 창 목록 (회차는 원래 값)"""
    left, right = _numeric(df, "left_vib", data, fname), _numeric(df, "right_vib", data, fname)
    if left is None and right is None:
        raise UserDataError(f"{fname}: 진동 열이 없습니다. left_vib 또는 right_vib가 필요합니다. 파일의 열: {list(df.columns)}")
    one_side = left is None or right is None
    if one_side:
        data.warn(f"{fname}: 진동이 한쪽뿐이라 같은 값을 양쪽에 넣었습니다. 좌우 비교 없이 판단하며 바퀴 위치는 알 수 없습니다.")
        left = right = left if right is None else right
    n = len(df)
    if "time" in df.columns:
        t = _seconds(df["time"], fname)
        if np.isnan(t).any():
            raise UserDataError(f"{fname}: time 열에 빈칸이 있습니다.")
        if (np.diff(t) < 0).any():
            raise UserDataError(f"{fname}: time이 줄어드는 곳이 {int((np.diff(t) < 0).sum())}군데 있습니다. "
                                "기록이 다시 시작됐다면 파일을 나눠 주세요.")
        steps = np.diff(t[~np.isnan(left)])
        steps = steps[steps > 0]
        if not len(steps):
            raise UserDataError(f"{fname}: time 값이 모두 같아 샘플링 주파수를 알 수 없습니다.")
        fs = 1.0 / float(np.median(steps))
    else:
        fs = float(fs_arg or C.SENSORS["vib_fs"])
        t = np.arange(n) / fs
        data.warn(f"{fname}: time 열이 없어 진동 샘플링을 {fs:g} Hz로 가정했습니다.")
    if fs < C.USER_MIN_FS:
        raise UserDataError(f"{fname}: 진동 샘플링이 {fs:.1f} Hz로 계산됐습니다 ({C.USER_MIN_FS:g} Hz 이상 필요). "
                            "time 열의 단위가 초가 맞는지 확인하세요.")
    if fs < 300:
        data.warn(f"{fname}: 진동 샘플링 {fs:.0f} Hz는 낮아서 고주파(150 Hz 이상) 특징이 거의 비게 됩니다.")

    optional = {k: _numeric(df, k, data, fname) for k in ("left_cur", "right_cur", "left_temp", "right_temp", "speed", "lap", "fault_severity")}
    for key, what in (("cur", "전류"), ("temp", "온도")):
        a, b = optional[f"left_{key}"], optional[f"right_{key}"]
        if a is None and b is None:
            data.warn(f"{fname}: {what} 열이 없어 {what} 특징을 0으로 두었습니다. {what}로 알 수 있는 고장은 놓칠 수 있습니다.")
        elif a is None or b is None:
            optional[f"left_{key}"] = optional[f"right_{key}"] = a if b is None else b
    speed = optional["speed"]
    if speed is None:
        data.warn(f"{fname}: speed 열이 없어 {C.USER_SPEED:g} m/s로 가정했습니다. 회전 주파수(1X) 특징과 전류 잔차의 의미가 약해집니다.")
    laps = optional["lap"]
    if laps is None:
        data.warn(f"{fname}: lap 열이 없어 {lap_s:g}초를 1회차로 나눴습니다.")
    seg = df["segment"].fillna(C.USER_SEGMENT).astype(str).str.strip().to_numpy() if "segment" in df.columns else None
    if seg is None:
        data.warn(f"{fname}: segment 열이 없어 전체를 한 구간으로 보고 기준을 만듭니다. 노면 차이는 걸러지지 않습니다.")
    bump = df["bump"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t", "y", "yes"]).to_numpy() if "bump" in df.columns else None
    kinds = _labels(df["fault_kind"], KIND_NAMES) if "fault_kind" in df.columns else None
    wheels = _labels(df["fault_wheel"], WHEEL_NAMES) if "fault_wheel" in df.columns else None
    sevs = optional["fault_severity"]
    has_label = kinds is not None or sevs is not None

    idx = np.floor((t - t[0]) / C.WINDOW_S + 1e-9).astype(int)
    bounds = np.flatnonzero(np.diff(idx)) + 1
    need, prev_speed, dropped, recs = C.USER_MIN_WINDOW_FILL * fs * C.WINDOW_S, None, 0, []
    last = {k: (0.0 if optional[k] is None else float(optional[k][~np.isnan(optional[k])][0])) for k in ("left_cur", "right_cur", "left_temp", "right_temp")}
    last_speed = C.USER_SPEED if speed is None else float(speed[~np.isnan(speed)][0])
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):
        lv, rv = left[a:b], right[a:b]
        lv, rv = lv[~np.isnan(lv)], rv[~np.isnan(rv)]
        if min(len(lv), len(rv)) < need:
            dropped += 1
            continue
        for k in last:
            if optional[k] is not None:
                last[k] = _mean(optional[k][a:b], last[k])
        last_speed = _mean(speed[a:b], last_speed) if speed is not None else last_speed
        accel = 0.0 if prev_speed is None else (last_speed - prev_speed) / C.WINDOW_S
        prev_speed = last_speed
        lap_vals = laps[a:b][~np.isnan(laps[a:b])] if laps is not None else ()
        rec = dict(lap=lap_vals[0] if len(lap_vals) else int((t[a] - t[0]) // lap_s), segment=seg[a] if seg is not None else C.USER_SEGMENT,
                   speed=last_speed, accel=accel, bump=bool(bump[a:b].any()) if bump is not None else False, vib_fs=fs,
                   left_vib=lv.astype(np.float32), right_vib=rv.astype(np.float32), left_cur=last["left_cur"], right_cur=last["right_cur"],
                   left_temp=last["left_temp"], right_temp=last["right_temp"], labeled=False, true_kind="normal", true_side="", true_sev=0.0)
        if has_label:
            kind = next((k for k in (kinds[a:b][::-1] if kinds is not None else ()) if k), "")
            sv = sevs[a:b][~np.isnan(sevs[a:b])] if sevs is not None else ()
            sev = float(np.clip(np.max(sv), 0, 1)) if len(sv) else None
            if not kind and sev:
                kind = "fault"                             # 고장 정도만 있고 종류가 없는 경우
            if kind or sev is not None:
                sev = 0.0 if kind in ("", "normal") else (1.0 if sev is None else sev)
                side = "" if one_side or wheels is None else next((w for w in wheels[a:b][::-1] if w in C.SIDE_KO), "")
                rec.update(labeled=True, true_kind=kind if sev > 0 else "normal", true_side=side if sev > 0 else "", true_sev=sev)
        recs.append(rec)
    if dropped:
        data.warn(f"{fname}: 샘플이 부족한 1초 창 {dropped}개를 버렸습니다.")
    return recs, fs


def load(path=C.USER_DATA_DIR, fs=None, lap_s=None):
    """파일 또는 폴더 → UserData. 파일 하나가 로봇 1대이고, robot 열이 있으면 그 값으로 나누거나 합친다."""
    data, lap_s = UserData(), float(lap_s or C.USER_LAP_S)
    files = path if isinstance(path, (list, tuple)) else list_files(path)
    if any(Path(f).name.startswith(SAMPLE_PREFIX) for f in files):
        data.warn(f"{SAMPLE_PREFIX}로 시작하는 파일은 시뮬레이터로 만든 예시입니다. 실제 데이터를 넣을 때는 지우세요.")
    for f in files:
        f = Path(f)
        df = read_table(f)
        stem = f.name.split(".")[0]
        groups = df.groupby(df["robot"].astype(str), sort=False) if "robot" in df.columns else [(stem, df)]
        for robot, part in groups:
            recs, rate = _windows(part, f.name, data, fs, lap_s)
            have = data.robots.setdefault(str(robot), [])
            offset = have[-1]["lap"] + 1 if have else 0    # 같은 로봇의 다음 파일은 이어지는 회차로 붙임
            order = {v: i for i, v in enumerate(dict.fromkeys(r["lap"] for r in recs))}
            for r in recs:
                r["lap"] = offset + order[r["lap"]]
            have += recs
            data.fs[str(robot)] = rate
    for robot, recs in data.robots.items():
        n_laps = recs[-1]["lap"] + 1 if recs else 0
        if n_laps <= C.BASELINE_LAPS:
            data.warn(f"{robot}: {n_laps}회차뿐이라 진단할 수 없습니다. 처음 {C.BASELINE_LAPS}회차로 기준을 만들고 그 뒤부터 진단합니다. "
                      "회차를 짧게 나누려면 1회차 길이를 줄이세요.")
        if any(r["true_sev"] > 0 and r["lap"] < C.BASELINE_LAPS for r in recs):
            data.warn(f"{robot}: 처음 {C.BASELINE_LAPS}회차에 고장 라벨이 있습니다. 기준 구간은 정상이어야 하므로 "
                      "정상 기록을 앞에 붙이세요 (robot 열을 같게 하면 파일을 이어 붙입니다).")
    return data


# ---------------------------------------------------------------- 진단
def diagnose(pipe, recs, detector="ae"):
    """로봇 1대를 처음부터 끝까지 진단 → (회차별 표, 추론기)"""
    if not recs or recs[-1]["lap"] < C.BASELINE_LAPS:
        raise UserDataError(f"회차가 부족합니다. 진단하려면 {C.BASELINE_LAPS + 1}회차 이상이 필요합니다.")
    mon, rows, last, truth = pipe.monitor(detector), [], None, {}

    def flush():
        for lap, score in mon.lap_scores[len(rows):]:
            fc = mon.forecast
            label = truth.get(lap)
            rows.append({"회차": lap, "건강 점수": round(score, 2), "상태": C.STATUS_KO[mon.status],
                         "정지 위험까지(회차)": round(fc["laps"], 1) if fc else None,
                         "빠르면(회차)": round(fc["laps_lo"], 1) if fc else None,
                         "추정 원인": (last or {}).get("cause") or "", "판단 근거": ", ".join(n for n, _ in mon.reason),
                         "파일 라벨": label or ""})

    for rec in recs:
        o = mon.update(rec)
        if rec["labeled"]:
            kind = C.KIND_KO.get(rec["true_kind"], rec["true_kind"])
            truth[rec["lap"]] = f"{kind} {rec['true_sev']:.2f}" if rec["true_sev"] > 0 else kind
        flush()
        if o["phase"] == "monitor":
            last = o
    mon.finish()
    flush()
    return pd.DataFrame(rows), mon


# ---------------------------------------------------------------- 학습
def _rows(recs, robot):
    rows = [extract(r) for r in recs]
    for r in rows:
        r["robot"] = robot
    return rows


def train_user(data, holdout=()):
    """내 데이터로 파이프라인 학습 → (파이프라인, 안내 문구 목록). holdout 로봇은 학습에서 빼고 성능 확인에 쓴다."""
    notes, normal, fault, skipped = [], [], [], []
    for name, recs in data.robots.items():
        if name in holdout:
            continue
        rows = _rows(recs, name)
        if not any(r["true_sev"] > 0 for r in rows):
            normal += rows
        elif any(r["true_sev"] > 0 and r["lap"] < C.BASELINE_LAPS for r in rows):
            skipped.append(name)
        else:
            fault += rows
    if skipped:
        notes.append(f"처음 {C.BASELINE_LAPS}회차에 고장 라벨이 있어 학습에서 뺀 로봇: {', '.join(skipped)}")
    if not any(r["labeled"] for name in data.robots if name not in holdout for r in data.robots[name]):
        notes.append("라벨이 없어 모든 로봇을 정상으로 보고 학습했습니다. 고장 기록이 섞여 있으면 고장을 정상으로 배우게 됩니다.")
    if not normal:                                         # 정상 로봇이 따로 없으면 고장 로봇의 고장 전 구간을 씀
        normal = [r for r in fault if r["true_sev"] == 0]
    n_use = sum(1 for r in normal if r["lap"] >= C.BASELINE_LAPS and not r["bump"])
    if n_use < C.USER_MIN_TRAIN_WINDOWS:
        raise UserDataError(f"학습에 쓸 정상 창이 {n_use}개뿐입니다. 처음 {C.BASELINE_LAPS}회차를 뺀 정상 구간이 "
                            f"{C.USER_MIN_TRAIN_WINDOWS}초 이상 필요합니다.")

    pipe = AMRPipeline().fit(dict(train_normal=normal, train_fault=fault), fit_robots=None, cal_robots=None)
    b = pipe.bundle
    b.update(source="user", trained_robots=sorted({r["robot"] for r in normal + fault}))
    ae = b["detectors"]["ae"]
    notes.append(f"학습 로봇 {len(b['trained_robots'])}대, 정상 창 {n_use:,}개, 고장 창 {sum(r['true_sev'] > 0 for r in fault):,}개")
    if b["scale"] == "normal_only":
        notes.append("고장 데이터가 없어 건강 점수는 정상 범위만 기준으로 삼습니다. 3점은 정상의 99.5% 지점이고, 정지 위험 판정과 남은 회차 예측은 하지 않습니다.")
    elif b["scale"] == "all_fault":
        notes.append(f"고장 정도 {C.FAIL_CAL_BAND[0]}~{C.FAIL_CAL_BAND[1]} 구간의 창이 없어 고장 창 전체의 중앙값을 9점으로 맞췄습니다.")
    if b["scale"] != "normal_only" and ae.rfail <= ae.r995:
        notes.append("고장 창의 이상 점수가 정상 범위와 구분되지 않습니다. 9점 눈금을 맞추지 못했으니 데이터와 라벨을 확인하세요.")
    if b["cause"] is None:
        notes.append("고장 라벨이 없어 원인 분류 모델은 만들지 않았습니다. 이상 여부와 건강 점수만 나옵니다.")
    else:
        notes.append(f"원인 분류 항목: {', '.join(label_ko(c) for c in b['cause'].classes)}")
    notes.append("학습에 쓴 로봇의 진단 결과는 실제 성능이 아닙니다. 성능은 학습에서 뺀 로봇으로 확인하세요.")
    b["notes"] = notes
    return pipe, notes


# ---------------------------------------------------------------- 예시 파일 (시뮬레이터로 생성)
def write_sample(path, faults=(), n_laps=30, seed=700):
    """이 형식을 그대로 따르는 CSV를 만든다. 전류·온도처럼 느린 센서는 값이 있는 줄에만 채운다."""
    from sim.simulator import AMRSimulator
    parts, sim = [], AMRSimulator(faults=list(faults), n_laps=n_laps, seed=seed)
    for i, rec in enumerate(sim.windows()):
        n = len(rec["left_vib"])
        step = n // len(rec["left_cur"])
        part = {"time": np.round(i * C.WINDOW_S + np.arange(n) / n * C.WINDOW_S, 4), "lap": rec["lap"], "segment": rec["segment"],
                "speed": round(rec["speed"], 4), "bump": int(rec["bump"]),
                "left_vib": np.round(rec["left_vib"], 5), "right_vib": np.round(rec["right_vib"], 5)}
        for key in ("left_cur", "right_cur"):
            col = np.full(n, np.nan)
            col[::step] = np.round(rec[key], 4)[:len(col[::step])]
            part[key] = col
        for key in ("left_temp", "right_temp"):
            col = np.full(n, np.nan)
            col[0] = rec[key]
            part[key] = col
        label = next(((k, w, rec[f"{w}_true_{k}"]) for w in ("left", "right") for k in C.FAULT_KINDS if rec[f"{w}_true_{k}"] > 0), None)
        part.update(fault_kind=label[0] if label else "normal", fault_wheel=label[1] if label else "",
                    fault_severity=round(label[2], 4) if label else 0.0)
        parts.append(pd.DataFrame(part))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return path
