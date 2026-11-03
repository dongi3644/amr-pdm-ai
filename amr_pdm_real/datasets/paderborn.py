"""Paderborn 대학 베어링 데이터 (KAt-DataCenter) → 이 프로젝트의 CSV 형식

[역할] 공개 데이터의 .mat 파일을 dataio.py가 읽는 CSV로 바꾸는 '어댑터'. 파이프라인 본체(pipeline/)는 공개 데이터를 모른다.
       직접 모은 데이터를 쓸 때는 이 파일이 필요 없다. 다른 공개 데이터를 붙일 때도 같은 모양의 변환기를 하나 더 만들면 된다.
[변환 내용] 진동 64 kHz → 8 kHz (저역통과 후 다운샘플), 상전류 2개 → 10 ms마다 RMS, 회전수 → 속도 열 (온도는 버림, 배터리 없음),
           측정 80회(각 4초)를 시간순으로 이어 붙임, 운전 조건 → 구간, 베어링 코드 → 로봇 이름·고장 종류 라벨
           판정 주기를 측정 1회에 맞추려면 학습할 때 --period-s 4 를 쓴다 (기본 60초면 측정 15회가 한 주기)

    python run.py paderborn download --bearings K001,K002,KA04     # raw/ 에 내려받고 압축 해제
    python run.py paderborn convert                                # raw/ 의 .mat → data/paderborn_*.csv.gz

출처: Lessmeier et al., "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using
Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification", PHM Society
European Conference, 2016. 라이선스 CC BY-NC 4.0 (비상업 학술 이용 가능, 출처 표기 필수).

데이터 구성 (베어링 1개 = 이 프로젝트의 '로봇' 1대)
  측정 1회 = 4초, 운전 조건 4가지 × 20회 = 80회 (총 약 5분 20초), 운전 조건 = '구간'
  진동 vibration_1 64 kHz, 모터 상전류 phase_current_1·2 64 kHz, 회전수 speed 4 kHz, 베어링 온도 1 Hz
  K = 정상, KA = 외륜 손상, KI = 내륜 손상, KB = 외륜+내륜 손상
  아래 REAL_DAMAGE는 가속 수명 시험으로 생긴 실제 손상, ARTIFICIAL은 가공으로 만든 인공 손상이다.
  같은 베어링의 측정은 반드시 같은 쪽(학습 또는 평가)에 둬야 한다 (workflow.cross_validate가 지킴).
"""
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{}.rar"
HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
REAL_DAMAGE = ["KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
ARTIFICIAL = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09", "KI01", "KI03", "KI05", "KI07", "KI08"]
KIND = {"K0": "normal", "KA": "outer_race", "KI": "inner_race", "KB": "combined"}
RAW_FS = 64000
CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2", "speed")      # 쓰는 신호 (온도·토크·하중은 쓰지 않음)
NAME_RE = re.compile(r"^(N\d+_M\d+_F\d+)_(K[A-Z0-9]\d+)_(\d+)\.mat$", re.I)


def kind_of(code):
    return KIND[code[:2].upper()]


# ---------------------------------------------------------------- 내려받기
def download(codes, raw_dir=C.RAW_DIR):
    """공식 서버에서 베어링별 .rar를 받아 raw/<코드>/ 에 푼다 (이미 있으면 건너뜀). 풀기는 Windows 기본 tar(bsdtar)를 쓴다."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for code in [c.strip().upper() for c in codes if c.strip()]:
        if code not in HEALTHY + REAL_DAMAGE + ARTIFICIAL:
            raise ValueError(f"알 수 없는 베어링 코드: {code}")
        if list((raw_dir / code).glob("*.mat")):
            print(f"{code}: 이미 있음")
            continue
        archive = raw_dir / f"{code}.rar"
        if not archive.exists():
            print(f"{code}: 내려받는 중 {URL.format(code)}", flush=True)
            tmp = archive.with_suffix(".part")
            with urllib.request.urlopen(URL.format(code), timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(archive)
        print(f"{code}: 압축 푸는 중", flush=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(raw_dir)], check=True)
        if not list((raw_dir / code).glob("*.mat")):
            raise RuntimeError(f"{code}: 압축을 풀었지만 {raw_dir / code} 에 .mat 파일이 없습니다.")
        archive.unlink()                                   # 풀고 나면 압축 파일은 지워 용량을 아낌


# ---------------------------------------------------------------- .mat 읽기
def _text(x):
    a = np.asarray(x).ravel()
    return str(a[0]) if a.size else ""


def _walk(obj, found):
    """.mat 구조체 안을 돌며 Name·Data 필드를 가진 항목을 신호 이름 → 배열로 모은다 (필드 위치에 의존하지 않음)"""
    if isinstance(obj, np.void) and obj.dtype.names:
        names = obj.dtype.names
        if "Name" in names and "Data" in names:
            found.setdefault(_text(obj["Name"]), np.asarray(obj["Data"], float).ravel())
        for n in names:
            _walk(obj[n], found)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names or obj.dtype == object:
            for item in obj.flat:
                _walk(item, found)


def read_mat(path):
    from scipy.io import loadmat
    found = {}
    for key, val in loadmat(path).items():
        if not key.startswith("__"):
            _walk(val, found)
    missing = [c for c in ("vibration_1", "speed") if c not in found]
    if missing:
        raise ValueError(f"{Path(path).name}: 신호 {missing}를 찾지 못했습니다. 있는 신호: {sorted(found)}")
    return found


# ---------------------------------------------------------------- 변환
def convert_bearing(code, fs=8000, raw_dir=C.RAW_DIR, out_dir=C.DATA_DIR):
    """베어링 1개의 측정 80회 → CSV 1개. 순서는 (반복 번호, 운전 조건)이라 측정 4회마다 모든 조건이 한 번씩 나온다."""
    from scipy.signal import resample_poly
    files = []
    for p in (raw_dir / code).glob("*.mat"):
        m = NAME_RE.match(p.name)
        if m:
            files.append((int(m.group(3)), m.group(1), p))
    if not files:
        raise FileNotFoundError(f"{raw_dir / code} 에 .mat 파일이 없습니다. 먼저 download 하세요.")
    parts, t0, skipped = [], 0.0, []
    kind = kind_of(code)
    for period, (run, cond, p) in enumerate(sorted(files)):
        try:
            s = read_mat(p)
        except Exception as e:                             # 공식 데이터에도 읽히지 않는 파일이 있음 (예: KA08 N15_M01_F10_2)
            skipped.append(f"{p.name}: {e}")
            continue
        vib = resample_poly(s["vibration_1"], fs, RAW_FS)  # 위상 지연 없는 FIR 저역통과 후 다운샘플 (앨리어싱 방지)
        n = len(vib)
        dur = n / fs
        row_t = t0 + np.arange(n) / fs
        cur = np.full(n, np.nan)                           # 상전류는 교류라 10 ms마다 RMS로 바꿔 넣음 (두 상 평균)
        step_raw, step = RAW_FS // 100, fs // 100
        phases = [s[c] for c in ("phase_current_1", "phase_current_2") if c in s]
        if phases:
            k = min(len(ph) for ph in phases) // step_raw
            rms = np.mean([np.sqrt(np.mean(ph[:k * step_raw].reshape(k, step_raw) ** 2, axis=1)) for ph in phases], axis=0)
            idx = np.arange(k) * step
            cur[idx[idx < n]] = rms[:np.sum(idx < n)]
        rpm = float(np.mean(s["speed"]))
        parts.append(pd.DataFrame({
            "time": np.round(row_t, 6), "robot": code, "period": period, "segment": cond,
            "speed": round(rpm / 60 * 2 * np.pi * C.WHEEL_RADIUS, 5),   # 회전수 → '바퀴 반지름 기준 속도' (1X 주파수 = rpm/60)
            # 센서가 하나라 첫 번째 바퀴(config.WHEELS[0])에 넣음. 온도는 이 프로젝트에서 쓰지 않음, 배터리 정보는 없음
            f"{C.WHEELS[0]}_vib": np.round(vib, 6), f"{C.WHEELS[0]}_cur": cur,
            "fault_kind": kind}))
        t0 += np.ceil(dur)                                 # 측정 사이를 1초 경계에 맞춰 창이 두 측정에 걸치지 않게 함
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"paderborn_{code}.csv.gz"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    return out, len(parts), skipped


def convert_all(codes=None, fs=8000, raw_dir=C.RAW_DIR):
    codes = codes or sorted(p.name for p in raw_dir.iterdir() if p.is_dir() and list(p.glob("*.mat")))
    if not codes:
        raise FileNotFoundError(f"{raw_dir} 에 내려받은 베어링이 없습니다. 먼저 python run.py paderborn download --bearings ... 를 실행하세요.")
    for code in codes:
        out, n, skipped = convert_bearing(code.strip().upper(), fs, raw_dir)
        print(f"{code}: 측정 {n}회 → {out.name}" + (f" (읽지 못한 파일 {len(skipped)}개: {skipped})" if skipped else ""), flush=True)
