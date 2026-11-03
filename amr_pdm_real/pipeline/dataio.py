"""실측 데이터 읽기: CSV → 파이프라인이 먹는 1초 창

[역할] 파이프라인의 맨 앞. 사용자가 넣은 CSV를 읽어 검사하고, 모든 모델이 같은 모양으로 받을 수 있게 '1초 창'으로 자른다.
       형식이 조금 달라도(열 이름, 인코딩, 날짜 형식, 일부 바퀴 센서 없음) 최대한 맞춰 읽고, 가정한 내용은 warnings에 남긴다.
[흐름에서 위치] CSV 파일 → (여기) → features.extract → 모델

    data = load("data")          # 파일 또는 폴더. 형식 설명은 README.md
    data.robots["R1"]            # 로봇(베어링) 1대의 1초 창 목록
    data.summary(), data.warnings

CSV 한 줄 = 진동 샘플 하나. 파일 하나 = 로봇 1대 (robot 열이 있으면 그 값으로 나누거나 이어 붙인다).

[1초 창 하나(rec)의 내용]
    period, segment, speed, accel, bump     판정 주기 번호(기록 시작부터 PERIOD_S초마다 1씩)·구간·속도·가속도·문턱 여부
    vib_fs                                  진동 샘플링 [Hz] (time 열에서 계산)
    <바퀴>_vib  (fl_vib, fr_vib, ...)       바퀴마다 그 1초 동안의 진동 파형 (배열). 바퀴 목록은 config.WHEELS
    <바퀴>_cur  (fl_cur, fr_cur, ...)       바퀴마다 그 1초의 전류 대표값 (숫자)
    battery, bat_drain                      배터리 잔량 [%], 최근 BATTERY_DROPS칸이 떨어진 시간으로 구한 소모 속도 [%/분]
    t                                       창 시작 시각 [s]
    labeled, true_kind, true_side, true_sev 파일의 라벨 (평가용, 모델 입력에는 쓰지 않음)

[주요 함수]
    load()            파일들을 읽어 Dataset을 만든다 (같은 로봇의 여러 파일은 시각·주기 번호를 이어 붙임)
    _windows()        로봇 1대의 표를 1초 창으로 자르는 핵심 부분
    _battery_drain()  창마다 배터리 소모 속도 계산, 충전 중인 창 표시
    choose_baseline() 정상 기준 방식(per_unit / reference) 자동 결정
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

EXTENSIONS = (".csv", ".csv.gz", ".zip")
KIND_NAMES = {"healthy": "normal", "ok": "normal", "none": "normal", "정상": "normal",
              **{v: k for k, v in C.KIND_KO.items()}}
# fault_wheel 라벨에 적힌 이름 → config.WHEELS의 바퀴 이름 (예: '앞 오른쪽', 'front-right', 'fr' → 'fr')
WHEEL_NAMES = {name: w for w in C.WHEELS
               for name in [w, C.WHEEL_KO.get(w, w), C.WHEEL_KO.get(w, w).replace(" ", "")] + C.WHEEL_ALIASES.get(w, [])}


def wheel_column_aliases():
    """바퀴별 진동·전류 열의 다른 이름 목록을 만든다. 예) fl_vib ← vib_fl, fl_vibration, frontleft_vib, 앞왼쪽진동
    진동·전류 열이 하나뿐이면(vib, current) 첫 번째 바퀴로 본다."""
    out = {}
    for i, w in enumerate(C.WHEELS):
        names = [w] + C.WHEEL_ALIASES.get(w, [])
        out[f"{w}_vib"] = [p for a in names for p in (f"{a}_vib", f"vib_{a}", f"{a}_vibration", f"vibration_{a}", f"{a}진동")] + (["vib", "vibration", "진동"] if i == 0 else [])
        out[f"{w}_cur"] = [p for a in names for p in (f"{a}_cur", f"cur_{a}", f"{a}_current", f"current_{a}", f"{a}전류")] + (["cur", "current", "전류"] if i == 0 else [])
    return out


class DataError(ValueError):
    """사용자가 고칠 수 있는 데이터 문제. 메시지를 그대로 보여 준다."""


class Dataset:
    """읽은 결과 묶음. robots: 로봇 이름 → 1초 창 목록, fs: 로봇 이름 → 진동 샘플링, warnings: 확인할 점"""

    def __init__(self):
        self.robots, self.warnings, self.fs = {}, [], {}

    def warn(self, text):
        if text not in self.warnings:
            self.warnings.append(text)

    def label_of(self, name):
        """로봇의 대표 라벨: 고장 라벨이 하나라도 있으면 그 종류, 라벨이 정상뿐이면 normal, 라벨이 없으면 None"""
        recs = self.robots[name]
        kinds = [f"{r['true_kind']}|{r['true_side']}" for r in recs if r["labeled"] and r["true_sev"] > 0]
        if kinds:
            return max(set(kinds), key=kinds.count)
        return "normal|" if any(r["labeled"] for r in recs) else None

    def summary(self):
        rows = []
        for name, recs in self.robots.items():
            lbl = self.label_of(name)
            kind = "없음" if lbl is None else C.KIND_KO.get(lbl.split("|")[0], lbl.split("|")[0])
            rows.append({"로봇": name, "기록(분)": round(C.minutes(recs[-1]["period"] + 1), 1) if recs else 0, "1초 창": len(recs),
                         "진동 샘플링(Hz)": round(self.fs.get(name, 0), 1), "구간": len({r["segment"] for r in recs}), "라벨": kind})
        return pd.DataFrame(rows)


def list_files(path=C.DATA_DIR):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise DataError(f"경로를 찾을 수 없습니다: {path}")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith(EXTENSIONS))
    if not files:
        raise DataError(f"{path} 폴더에 CSV 파일이 없습니다. 형식은 README.md를 보세요.")
    return files


def read_table(path):
    """CSV 하나를 표로 읽고, 열 이름을 소문자로 바꾼 뒤 표준 이름(time, fl_vib ...)에 맞춘다 (config.COLUMN_ALIASES + 바퀴 별칭)."""
    for enc in ("utf-8-sig", "cp949"):                     # 엑셀에서 저장한 한글 CSV는 cp949일 수 있음
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError) as e:
            raise DataError(f"{Path(path).name}: CSV로 읽지 못했습니다 ({e})") from e
    else:
        raise DataError(f"{Path(path).name}: 글자 인코딩을 알 수 없습니다. UTF-8로 저장해 주세요.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for name, aliases in {**C.COLUMN_ALIASES, **wheel_column_aliases()}.items():
        if name not in df.columns:
            found = next((a for a in aliases if a in df.columns), None)
            if found:
                rename[found] = name
    return df.rename(columns=rename)


def _numeric(df, name, data, fname):
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
    raise DataError(f"{fname}: time 열을 시간으로 읽지 못했습니다. 초 단위 숫자나 날짜·시각 형식이어야 합니다.")


def _mean(values, prev, ac=False):
    v = values[~np.isnan(values)]
    if not len(v):
        return prev
    return float(np.sqrt(np.mean(v ** 2))) if ac else float(v.mean())


def _labels(col, names):
    return col.fillna("").astype(str).str.strip().str.lower().map(lambda s: names.get(s, s)).to_numpy()


def _windows(df, fname, data, fs_arg):
    """로봇 1대의 표 → 1초 창 목록 (판정 주기 번호 = 기록 시작부터 몇 번째 PERIOD_S초인가)

    순서: ① 바퀴별 진동 열 확인(없는 바퀴는 다른 바퀴 값으로 채움) ② time으로 샘플링 주파수 계산
          ③ 선택 열(전류·배터리·속도·구간·라벨) 준비 ④ time을 1초 단위로 잘라 창마다 진동 파형과 대표값을 담음
          ⑤ 샘플이 모자란 창은 버림"""
    # ① 진동: 바퀴끼리 비교(dev_ 특징)가 이 시스템의 핵심이라, 센서가 없는 바퀴는 있는 바퀴 값을 넣어 차이를 0으로 만든다
    vib = {w: _numeric(df, f"{w}_vib", data, fname) for w in C.WHEELS}
    measured = [w for w in C.WHEELS if vib[w] is not None]
    if not measured:
        raise DataError(f"{fname}: 진동 열이 없습니다. {', '.join(w + '_vib' for w in C.WHEELS)} 중 하나 이상이 필요합니다 "
                        f"(바퀴 수·이름은 config.WHEELS). 파일의 열: {list(df.columns)}")
    missing = [w for w in C.WHEELS if vib[w] is None]
    if missing:
        data.warn(f"{fname}: {', '.join(C.WHEEL_KO.get(w, w) for w in missing)} 바퀴 진동 열이 없어 "
                  f"{C.WHEEL_KO.get(measured[0], measured[0])} 바퀴 값을 넣었습니다. 이 바퀴들의 고장 위치는 구분할 수 없습니다.")
        for w in missing:
            vib[w] = vib[measured[0]]
    base = vib[measured[0]]
    # ② 샘플링 주파수 = 1 / (진동 샘플 사이 시간 간격의 중앙값). 단위가 초가 아니면 비정상적인 값이 나와 걸러진다
    n = len(df)
    if "time" in df.columns:
        t = _seconds(df["time"], fname)
        if np.isnan(t).any():
            raise DataError(f"{fname}: time 열에 빈칸이 있습니다.")
        if (np.diff(t) < 0).any():
            raise DataError(f"{fname}: time이 줄어드는 곳이 {int((np.diff(t) < 0).sum())}군데 있습니다. "
                            "기록이 다시 시작됐다면 파일을 나눠 주세요.")
        steps = np.diff(t[~np.isnan(base)])
        steps = steps[steps > 0]
        if not len(steps):
            raise DataError(f"{fname}: time 값이 모두 같아 샘플링 주파수를 알 수 없습니다.")
        fs = 1.0 / float(np.median(steps))
    else:
        fs = float(fs_arg or C.DEFAULT_FS)
        t = np.arange(n) / fs
        data.warn(f"{fname}: time 열이 없어 진동 샘플링을 {fs:g} Hz로 가정했습니다.")
    if fs < C.MIN_FS:
        raise DataError(f"{fname}: 진동 샘플링이 {fs:.1f} Hz로 계산됐습니다 ({C.MIN_FS:g} Hz 이상 필요). "
                        "time 열의 단위가 초가 맞는지 확인하세요.")
    if fs < 300:
        data.warn(f"{fname}: 진동 샘플링 {fs:.0f} Hz는 낮아서 고주파(150 Hz 이상) 특징이 거의 비게 됩니다.")

    # ③ 선택 열: 없으면 config의 기본값을 쓰고 경고를 남긴다. 느린 센서(전류·배터리)는 값이 있는 줄만 채워져 있어도 된다
    cur_keys = [f"{w}_cur" for w in C.WHEELS]
    opt = {k: _numeric(df, k, data, fname) for k in cur_keys + ["battery", "speed", "fault_severity"]}
    have_cur = [k for k in cur_keys if opt[k] is not None]
    if not have_cur:
        data.warn(f"{fname}: 전류 열이 없어 전류 특징을 0으로 두었습니다. 전류로 알 수 있는 고장은 놓칠 수 있습니다.")
    elif len(have_cur) < len(cur_keys):               # 모터가 없는 바퀴(캐스터 등)는 있는 바퀴 전류를 넣어 차이를 0으로
        data.warn(f"{fname}: 전류 열이 {', '.join(have_cur)}뿐이라 나머지 바퀴에는 {have_cur[0]} 값을 넣었습니다.")
        for k in cur_keys:
            if opt[k] is None:
                opt[k] = opt[have_cur[0]]
    ac = False
    if have_cur:                                       # 부호가 자주 바뀌면 교류 전류 원신호로 보고 창마다 RMS를 씀
        v = opt[have_cur[0]][~np.isnan(opt[have_cur[0]])]
        ac = len(v) > 100 and 0.2 < np.mean(v < 0) < 0.8
        if ac:
            data.warn(f"{fname}: 전류가 교류 원신호로 보여 1초 창마다 RMS로 바꿨습니다.")
    speed = opt["speed"]
    if speed is None:
        data.warn(f"{fname}: speed 열이 없어 {C.DEFAULT_SPEED:g} m/s로 가정했습니다. 회전 주파수(1X) 특징과 전류 잔차의 의미가 약해집니다.")
    seg =df["segment"].fillna(C.DEFAULT_SEGMENT).astype(str).str.strip().to_numpy() if "segment" in df.columns else None
    if seg is None:
        data.warn(f"{fname}: segment 열이 없어 전체를 한 구간으로 보고 기준을 만듭니다. 노면 차이는 걸러지지 않습니다.")
    bump = df["bump"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t", "y", "yes"]).to_numpy() if "bump" in df.columns else None
    kinds = _labels(df["fault_kind"], KIND_NAMES) if "fault_kind" in df.columns else None
    wheels = _labels(df["fault_wheel"], WHEEL_NAMES) if "fault_wheel" in df.columns else None
    sevs = opt["fault_severity"]
    has_label = kinds is not None or sevs is not None

    # ④ 창 나누기: 시작 시각 기준 몇 번째 1초인지(idx)가 바뀌는 지점이 창의 경계
    idx = np.floor((t - t[0]) / C.WINDOW_S + 1e-9).astype(int)
    bounds = np.flatnonzero(np.diff(idx)) + 1
    need, prev_speed, dropped, recs = C.MIN_WINDOW_FILL * fs * C.WINDOW_S, None, 0, []
    first = lambda a: float(a[~np.isnan(a)][0])
    last = {k: (0.0 if opt[k] is None else first(opt[k])) for k in cur_keys + ["battery"]}
    last_speed = C.DEFAULT_SPEED if speed is None else first(speed)
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):          # a:b = 이번 1초 창에 속한 줄 범위
        waves = {w: vib[w][a:b][~np.isnan(vib[w][a:b])] for w in C.WHEELS}
        if min(len(x) for x in waves.values()) < need:               # ⑤ 기록이 끊겨 샘플이 80% 미만이면 버림
            dropped += 1
            continue
        for k in last:                                               # 전류·배터리: 창 평균 (값이 없는 창은 직전 값 유지)
            if opt[k] is not None:
                last[k] = _mean(opt[k][a:b], last[k], ac=ac and "cur" in k)
        last_speed = _mean(speed[a:b], last_speed) if speed is not None else last_speed
        accel = 0.0 if prev_speed is None else (last_speed - prev_speed) / C.WINDOW_S
        prev_speed = last_speed
        rec = dict(period=int((t[a] - t[0]) // C.PERIOD_S),                # 판정 주기: 기록 시작부터 몇 번째 PERIOD_S초인가
                   segment=seg[a] if seg is not None else C.DEFAULT_SEGMENT,
                   speed=last_speed, accel=accel, bump=bool(bump[a:b].any()) if bump is not None else False, vib_fs=fs,
                   battery=last["battery"] if opt["battery"] is not None else None, t=float(t[a]),
                   labeled=False, true_kind="normal", true_side="", true_sev=0.0)
        for w in C.WHEELS:                                           # 바퀴마다 진동 파형과 전류 대표값
            rec[f"{w}_vib"] = waves[w].astype(np.float32)
            rec[f"{w}_cur"] = last[f"{w}_cur"]
        if has_label:                                                # 라벨: 창 안의 마지막 고장 종류와 최대 고장 정도
            kind = next((k for k in (kinds[a:b][::-1] if kinds is not None else ()) if k), "")
            sv = sevs[a:b][~np.isnan(sevs[a:b])] if sevs is not None else ()
            sev = float(np.clip(np.max(sv), 0, 1)) if len(sv) else None
            if not kind and sev:
                kind = "fault"                             # 고장 정도만 있고 종류가 없는 경우
            if kind or sev is not None:
                sev = 0.0 if kind in ("", "normal") else (1.0 if sev is None else sev)
                # 고장 바퀴: 센서가 있는 바퀴일 때만 위치를 라벨로 씀 (값을 빌려 넣은 바퀴는 위치를 구분할 수 없음)
                side = "" if wheels is None else next((w for w in wheels[a:b][::-1] if w in measured), "")
                rec.update(labeled=True, true_kind=kind if sev > 0 else "normal", true_side=side if sev > 0 else "", true_sev=sev)
        recs.append(rec)
    if dropped:
        data.warn(f"{fname}: 샘플이 부족한 1초 창 {dropped}개를 버렸습니다.")
    _battery_drain(recs, opt["battery"] is not None, data, fname)
    return recs, fs


def _battery_drain(recs, has_battery, data, fname):
    """창마다 배터리 소모 속도 [%/분] (클수록 빨리 닳음). 모델 특징과 '정상 소모' 학습 목표로 쓴다.

    잔량이 한 칸(BATTERY_STEP) 떨어진 시각들을 모아, 최근 BATTERY_DROPS칸이 떨어지는 데 걸린 시간으로 속도를 잰다.
        예) 3%가 떨어지는 데 9분 → 0.33 %/분
    칸이 바뀌는 시각은 정확하므로 1% 계단 때문에 흔들리지 않는다. 처음 몇 칸이 쌓이기 전은 처음 잰 값으로 채운다.
    잔량이 오르면 충전으로 보고 기록을 다시 시작하며, 그 뒤 다음 칸이 떨어질 때까지의 창은 bump처럼 판정에서 뺀다."""
    if not recs:
        return
    if not has_battery:
        for r in recs:
            r["bat_drain"] = 0.0
        data.warn(f"{fname}: battery 열이 없어 배터리 소모 이상은 판단하지 않습니다.")
        return
    drops, level, charging, n_charge = [], None, False, 0
    drain = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        lev = int(np.floor(r["battery"] / C.BATTERY_STEP + 1e-9))
        if level is not None and lev > level:              # 충전: 지난 소모 기록은 버리고 새로 시작
            drops, charging = [], True
            n_charge += 1
        elif level is not None and lev < level:            # 한 칸(이상) 떨어짐: 떨어진 칸 수만큼 시각을 기록
            drops += [r["t"]] * (level - lev)
            charging = False
        level = lev
        if charging:
            r["bump"] = True
        if len(drops) > C.BATTERY_DROPS:
            span = drops[-1] - drops[-1 - C.BATTERY_DROPS]
            if span > 0:
                drain[i] = C.BATTERY_DROPS * C.BATTERY_STEP / span * 60
    if np.isnan(drain).all():
        data.warn(f"{fname}: 잔량이 {C.BATTERY_DROPS + 1}칸 이상 떨어지지 않아 배터리 소모 속도를 잴 수 없습니다. 더 긴 기록이 필요합니다.")
        drain[:] = 0.0
    drain = pd.Series(drain).bfill().ffill().to_numpy()
    for r, d in zip(recs, drain):
        r["bat_drain"] = float(d)
    if n_charge:
        data.warn(f"{fname}: 충전(잔량 상승) {n_charge}번을 찾아 충전 직후 구간을 판정에서 뺐습니다.")


def load(path=C.DATA_DIR, fs=None):
    """파일 또는 폴더 → Dataset. 파일 하나가 로봇 1대이고, robot 열이 있으면 그 값으로 나누거나 합친다.
    판정 주기 번호는 '기록된 시간' 기준이라, 기록이 끊긴 시간(꺼져 있던 시간)은 건너뛰고 번호를 이어 매긴다."""
    data = Dataset()
    files = path if isinstance(path, (list, tuple)) else list_files(path)
    for f in files:
        f = Path(f)
        df = read_table(f)
        stem = f.name.split(".")[0]
        groups = df.groupby(df["robot"].astype(str), sort=False) if "robot" in df.columns else [(stem, df)]
        for robot, part in groups:
            recs, rate = _windows(part, f.name, data, fs)
            have = data.robots.setdefault(str(robot), [])
            offset = have[-1]["period"] + 1 if have else 0    # 같은 로봇의 다음 파일은 이어지는 주기 번호로 붙임
            order = {v: i for i, v in enumerate(dict.fromkeys(r["period"] for r in recs))}
            shift = (have[-1]["t"] + C.WINDOW_S - recs[0]["t"]) if (have and recs and recs[0]["t"] <= have[-1]["t"]) else 0.0
            for r in recs:
                r["period"] = offset + order[r["period"]]
                r["t"] += shift                            # 시각이 앞 파일보다 뒤로 가지 않게 (배터리 소모 계산이 시각 순서를 씀)
            have += recs
            data.fs[str(robot)] = rate
    return data


def choose_baseline(data, mode=None):
    """정상 기준 방식 결정. auto면 처음 BASELINE_PERIODS주기에 고장 라벨이 있거나 기록이 그보다 짧은 로봇이 있을 때 reference"""
    mode = mode or C.BASELINE_MODE
    if mode != "auto":
        return mode
    for recs in data.robots.values():
        if not recs or recs[-1]["period"] < C.BASELINE_PERIODS or any(r["true_sev"] > 0 and r["period"] < C.BASELINE_PERIODS for r in recs):
            return "reference"
    return "per_unit"
