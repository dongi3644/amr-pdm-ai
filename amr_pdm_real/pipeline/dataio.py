"""실측 데이터 읽기: CSV → 파이프라인이 먹는 1초 창 (진동만 사용)

[역할] 파이프라인의 맨 앞. 사용자가 넣은 CSV를 읽어 검사하고, 모든 모델이 같은 모양으로 받을 수 있게 '1초 창'으로 자른다.
       열 이름·인코딩·날짜 형식이 조금 달라도 맞춰 읽는다. 하지만 없는 데이터를 기본값이나 다른 바퀴 값으로
       채워 넣지는 않는다 — 필요한 값이 없으면 고칠 방법과 함께 오류를 낸다.
[흐름에서 위치] CSV 파일 → (여기) → features.extract → 모델

    data = load("data")          # 파일 또는 폴더. 형식 설명은 README.md
    data.robots["R1"]            # 로봇 1대의 1초 창 목록
    data.summary(), data.warnings

CSV 한 줄 = 진동 샘플 하나. 파일 하나 = 로봇 1대 (robot 열이 있으면 그 값으로 나누거나 이어 붙인다).

[1초 창 하나(rec)의 내용]
    period, segment, bump                   판정 주기 번호(기록 시작부터 PERIOD_S초마다 1씩)·구간·문턱 여부
    vib_fs                                  진동 샘플링 [Hz] (time 열에서 계산, 없으면 --fs로 직접 알려 줘야 함)
    <바퀴>_vib  (fl_vib, fr_vib, ...)       바퀴마다 그 1초 동안의 진동 파형 (배열). 바퀴 목록은 config.WHEELS
    t                                       창 시작 시각 [s]
    labeled, true_kind, true_side, true_sev 파일의 라벨 (평가용, 모델 입력에는 쓰지 않음)

[주요 함수]
    load()            파일들을 읽어 Dataset을 만든다 (같은 로봇의 여러 파일은 시각·주기 번호를 이어 붙임)
    _windows()        로봇 1대의 표를 1초 창으로 자르는 핵심 부분
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
    """바퀴별 진동 열의 다른 이름 목록을 만든다. 예) fl_vib ← vib_fl, fl_vibration, front_left_vib, 앞왼쪽진동"""
    out = {}
    for w in C.WHEELS:
        names = [w] + C.WHEEL_ALIASES.get(w, [])
        out[f"{w}_vib"] = [p for a in names for p in (f"{a}_vib", f"vib_{a}", f"{a}_vibration", f"vibration_{a}", f"{a}진동")]
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
        data.warn(f"{fname}: {name} 열에 숫자가 아닌 값 {bad}개를 빈칸으로 처리했습니다 (그 샘플은 쓰지 않음).")
    return col.to_numpy(float) if col.notna().any() else None


def _seconds(col, fname):
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.99:
        return num.to_numpy(float)
    dt = pd.to_datetime(col, errors="coerce")
    if dt.notna().mean() > 0.99:
        return (dt - dt.iloc[0]).dt.total_seconds().to_numpy(float)
    raise DataError(f"{fname}: time 열을 시간으로 읽지 못했습니다. 초 단위 숫자나 날짜·시각 형식이어야 합니다.")


def _labels(col, names):
    return col.fillna("").astype(str).str.strip().str.lower().map(lambda s: names.get(s, s)).to_numpy()


def _windows(df, fname, data, fs_arg):
    """로봇 1대의 표 → 1초 창 목록 (판정 주기 번호 = 기록 시작부터 몇 번째 PERIOD_S초인가)

    순서: ① 바퀴 4개 진동 열 확인(하나라도 없으면 오류) ② time으로 샘플링 주파수 계산
          ③ 선택 열(구간·문턱·라벨) 준비 ④ time을 1초 단위로 잘라 창마다 진동 파형을 담음 ⑤ 샘플이 모자란 창은 버림"""
    # ① 진동: 바퀴끼리 비교(dev_ 특징)가 이 시스템의 핵심이라 네 바퀴가 모두 있어야 한다. 빈 바퀴를 다른 값으로 채우지 않는다
    vib = {w: _numeric(df, f"{w}_vib", data, fname) for w in C.WHEELS}
    missing = [w for w in C.WHEELS if vib[w] is None]
    if missing:
        raise DataError(f"{fname}: {', '.join(f'{w}_vib({C.WHEEL_KO.get(w, w)})' for w in missing)} 열이 없습니다. "
                        f"바퀴 4개의 진동 열이 모두 필요합니다. 파일의 열: {list(df.columns)}")
    base = vib[C.WHEELS[0]]
    # ② 샘플링 주파수 = 1 / (진동 샘플 사이 시간 간격의 중앙값). time 열이 없으면 --fs로 알려 줘야 한다 (임의로 가정하지 않음)
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
    elif fs_arg:
        fs = float(fs_arg)
        t = np.arange(n) / fs
        data.warn(f"{fname}: time 열이 없어 지정한 샘플링 {fs:g} Hz로 시각을 계산했습니다.")
    else:
        raise DataError(f"{fname}: time 열이 없습니다. time 열을 넣거나, 센서의 실제 샘플링 주파수를 --fs 로 알려 주세요.")
    if fs < C.MIN_FS:
        raise DataError(f"{fname}: 진동 샘플링이 {fs:.1f} Hz로 계산됐습니다 ({C.MIN_FS:g} Hz 이상 필요). "
                        "time 열의 단위가 초가 맞는지 확인하세요.")
    if fs < 300:
        data.warn(f"{fname}: 진동 샘플링 {fs:.0f} Hz는 낮아서 고주파(150 Hz 이상) 특징이 거의 비게 됩니다.")

    # ③ 선택 열: 구간이 없으면 전체를 한 구간으로 본다 (값을 만들어 넣는 게 아니라 구간 구분을 하지 않는 것)
    seg = df["segment"].fillna(C.DEFAULT_SEGMENT).astype(str).str.strip().to_numpy() if "segment" in df.columns else None
    if seg is None:
        data.warn(f"{fname}: segment 열이 없어 전체를 한 구간으로 보고 기준을 만듭니다. 장소별 노면 차이는 걸러지지 않습니다.")
    bump = df["bump"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t", "y", "yes"]).to_numpy() if "bump" in df.columns else None
    kinds = _labels(df["fault_kind"], KIND_NAMES) if "fault_kind" in df.columns else None
    wheels = _labels(df["fault_wheel"], WHEEL_NAMES) if "fault_wheel" in df.columns else None
    sevs = _numeric(df, "fault_severity", data, fname)
    has_label = kinds is not None or sevs is not None

    # ④ 창 나누기: 시작 시각 기준 몇 번째 1초인지(idx)가 바뀌는 지점이 창의 경계
    idx = np.floor((t - t[0]) / C.WINDOW_S + 1e-9).astype(int)
    bounds = np.flatnonzero(np.diff(idx)) + 1
    need, dropped, recs = C.MIN_WINDOW_FILL * fs * C.WINDOW_S, 0, []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):          # a:b = 이번 1초 창에 속한 줄 범위
        waves = {w: vib[w][a:b][~np.isnan(vib[w][a:b])] for w in C.WHEELS}
        if min(len(x) for x in waves.values()) < need:               # ⑤ 기록이 끊겨 샘플이 80% 미만이면 버림
            dropped += 1
            continue
        rec = dict(period=int((t[a] - t[0]) // C.PERIOD_S),          # 판정 주기: 기록 시작부터 몇 번째 PERIOD_S초인가
                   segment=seg[a] if seg is not None else C.DEFAULT_SEGMENT,
                   bump=bool(bump[a:b].any()) if bump is not None else False, vib_fs=fs, t=float(t[a]),
                   labeled=False, true_kind="normal", true_side="", true_sev=0.0)
        for w in C.WHEELS:                                           # 바퀴마다 진동 파형
            rec[f"{w}_vib"] = waves[w].astype(np.float32)
        if has_label:                                                # 라벨: 창 안의 마지막 고장 종류와 최대 고장 정도
            kind = next((k for k in (kinds[a:b][::-1] if kinds is not None else ()) if k), "")
            sv = sevs[a:b][~np.isnan(sevs[a:b])] if sevs is not None else ()
            sev = float(np.clip(np.max(sv), 0, 1)) if len(sv) else None
            if not kind and sev:
                kind = "fault"                             # 고장 정도만 있고 종류가 없는 경우
            if kind or sev is not None:
                sev = 0.0 if kind in ("", "normal") else (1.0 if sev is None else sev)
                side = "" if wheels is None else next((w for w in wheels[a:b][::-1] if w in C.WHEELS), "")
                rec.update(labeled=True, true_kind=kind if sev > 0 else "normal", true_side=side if sev > 0 else "", true_sev=sev)
        recs.append(rec)
    if dropped:
        data.warn(f"{fname}: 샘플이 부족한 1초 창 {dropped}개를 버렸습니다.")
    return recs, fs


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
                r["t"] += shift                            # 시각이 앞 파일보다 뒤로 가지 않게
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
