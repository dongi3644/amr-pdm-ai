"""실측 데이터 읽기: CSV → 파이프라인이 먹는 1초 창

[역할] 파이프라인의 맨 앞. 사용자가 넣은 CSV를 읽어 검사하고, 모든 모델이 같은 모양으로 받을 수 있게 '1초 창'으로 자른다.
       열 이름·인코딩·날짜 형식이 조금 달라도 맞춰 읽는다. 하지만 없는 데이터를 기본값이나 다른 값으로
       채워 넣지는 않는다 — 필요한 값이 없으면 고칠 방법과 함께 오류를 낸다.
[흐름에서 위치] CSV 파일 → (여기) → features.extract → 진동 모델,  current·voltage → power.py (전력 감시)

    data = load("data")          # 파일 또는 폴더. 형식 설명은 README.md
    data.robots["R1"]            # 로봇 1대의 1초 창 목록
    data.summary(), data.warnings

CSV 한 줄 = 샘플 하나 (아두이노가 한 번 읽은 값). 파일 하나 = 로봇 1대 (robot 열이 있으면 그 값으로 나누거나 이어 붙인다).

[1초 창 하나(rec)의 내용]
    period, segment, bump                   판정 주기 번호(기록 시작부터 PERIOD_S초마다 1씩)·구간·문턱 여부
    vib_fs                                  진동 샘플링 [Hz] (time 열에서 계산, 없으면 --fs로 직접 알려 줘야 함)
    ax, ay, az, gx, gy, gz                  MPU6050 축마다 그 1초 동안의 파형 (배열). 목록은 config.VIB_CHANNELS
    current, voltage                        그 1초 동안 INA219 값의 평균 [A], [V]. 열이 없거나 그 1초에 값이 없으면 nan
    t                                       창 시작 시각 [s]
    labeled, true_kind, true_side, true_sev 파일의 라벨 (평가용, 모델 입력에는 쓰지 않음)

[주요 함수]
    load()            파일들을 읽어 Dataset을 만든다 (같은 로봇의 여러 파일은 시각·주기 번호를 이어 붙임)
    _windows()        로봇 1대의 표를 1초 창으로 자르는 핵심 부분
    choose_baseline() 정상 기준 방식(per_unit / reference) 자동 결정
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

EXTENSIONS = (".csv", ".csv.gz", ".zip")   # 이 확장자만 데이터로 읽는다 (data/README.md 같은 안내 파일은 무시됨)
# fault_kind 라벨에 적힌 이름 → 표준 이름. 'healthy', '정상' → 'normal', '베어링 결함' → 'bearing' 처럼
# 영어·한국어 어느 쪽으로 적어도 같은 고장으로 알아듣게 한다 (한국어 이름은 config.KIND_KO를 거꾸로 뒤집어 만듦)
KIND_NAMES = {"healthy": "normal", "ok": "normal", "none": "normal", "정상": "normal",
              "idle": "normal", "stop": "normal", "stopped": "normal", "정지": "normal", "대기": "normal",   # 멈춰 있는 정상 상태
              **{v: k for k, v in C.KIND_KO.items()}}
# fault_wheel 라벨에 적힌 이름 → config.WHEELS의 바퀴 이름 (예: '앞 오른쪽', 'front-right', 'fr' → 'fr')
WHEEL_NAMES = {name: w for w in C.WHEELS
               for name in [w, C.WHEEL_KO.get(w, w), C.WHEEL_KO.get(w, w).replace(" ", "")] + C.WHEEL_ALIASES.get(w, [])}


POWER_COLUMNS = ("current", "voltage")      # INA219 열 (선택). 진동보다 드물게 기록돼 빈칸이 많아도 됨
# 단위가 이름에 붙은 열 → (표준 열, 곱할 값). 예: mV → voltage[V] (× 0.001), t_us → time[s] (× 0.000001)
UNIT_COLUMNS = {"time": [("t_us", 1e-6), ("time_us", 1e-6), ("micros", 1e-6), ("t_ms", 1e-3), ("time_ms", 1e-3), ("millis", 1e-3)],
                "voltage": [("mv", 1e-3), ("voltage_mv", 1e-3), ("bus_mv", 1e-3)],
                "current": [("ma", 1e-3), ("current_ma", 1e-3)]}
# 엔코더·모터 명령 (선택 열). 창마다 바퀴 속도[펄스/초]와 모터 명령 평균을 담아 check·diagnose 표에 보여 준다 (모델 입력 아님)
AUX_COLUMNS = tuple(C.ENCODER_CHANNELS) + ("motor_cmd",)
# 파이프라인이 읽는 열 (나머지 열은 '쓰지 않은 열'로 한 번 알려 줌)
USED_COLUMNS = {"time", "robot", "segment", "bump", "current", "voltage", "fault_kind", "fault_wheel", "fault_severity", *AUX_COLUMNS}
# 열 이름 끝에 붙은 단위 → 표준 단위로 곱할 값. 예: '시간(ms)' → time × 0.001, '전류[mA]' → current × 0.001
UNIT_SUFFIX = {"us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0,
               "ma": 1e-3, "a": 1.0, "mv": 1e-3, "v": 1.0}


def _unwrap_counter(v, bits=32):
    """2^bits에서 0으로 돌아가는 정수 카운터를 이어지게 펴 줌 (크게 줄어들거나 늘어나는 곳마다 2^bits를 더하거나 뺌)
    bits=32: 아두이노 micros()/millis() 시각. bits=16: 아두이노 int(16비트)로 센 엔코더 펄스 (32767 다음에 -32768)"""
    v = np.asarray(v, float)
    d = np.r_[0, np.diff(v)]
    half = 2.0 ** (bits - 1)
    jumps = (d < -half).astype(float) - (d > half).astype(float)
    return v + np.cumsum(jumps) * 2.0 ** bits if jumps.any() else v


def _standardize(df, fname, data):
    """기록 스케치마다 다른 단위·형식을 표준(초, g, °/s, A, V)으로 맞춘다. 바꾼 것은 모두 '확인할 점'으로 알린다.
    - t_us·t_ms 같은 열 → time[s] (micros() 한 바퀴 돌기도 이어 붙임)
    - mV·mA 열 → voltage[V]·current[A]
    - 가속도가 센서 원래 정수값이면 (값이 50을 넘으면 g일 수 없음) config의 눈금으로 나눠 g·°/s로
    - 값이 모두 0인 전류·전압 열 → 센서를 못 읽은 것으로 보고 쓰지 않음 (0 V는 실제 값이 될 수 없음)"""
    df = df.copy()
    for target, cands in UNIT_COLUMNS.items():
        if target in df.columns:
            continue
        for col, scale in cands:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce")
                if target == "time":
                    v = pd.Series(_unwrap_counter(v.to_numpy(float)), index=df.index)
                df[target] = v * scale
                data.warn(f"{col} 열을 {target}[{'s' if target == 'time' else 'V' if target == 'voltage' else 'A'}]로 바꿔 읽었습니다 (× {scale:g}).")
                break
    acc = [ch for ch in ("ax", "ay", "az") if ch in df.columns]
    if acc:
        a = df[acc].apply(pd.to_numeric, errors="coerce")
        if np.nanmax(np.abs(a.to_numpy(float))) > 50:      # g 단위라면 MPU6050 최대 ±16 g를 넘을 수 없음 → 원래 정수값
            for ch in acc:
                df[ch] = a[ch] / C.RAW_ACCEL_LSB_PER_G
            gyr = [ch for ch in ("gx", "gy", "gz") if ch in df.columns]
            for ch in gyr:
                df[ch] = pd.to_numeric(df[ch], errors="coerce") / C.RAW_GYRO_LSB_PER_DPS
            data.warn(f"가속도{'·각속도' if gyr else ''}가 센서 원래값이라 가속도 ÷{C.RAW_ACCEL_LSB_PER_G:g}(g)"
                      + (f", 각속도 ÷{C.RAW_GYRO_LSB_PER_DPS:g}(°/s)" if gyr else "")
                      + "로 바꿨습니다. 스케치의 센서 범위 설정이 다르면 config.RAW_ACCEL_LSB_PER_G·RAW_GYRO_LSB_PER_DPS를 고치세요.")
        g = np.sqrt((df[acc].apply(pd.to_numeric, errors="coerce") ** 2).sum(axis=1)) if len(acc) == 3 else None
        if g is not None and g.notna().any() and not 0.7 < float(g.median()) < 1.3:   # 가속도 크기의 중앙값은 중력 1 g 근처여야 함
            data.warn(f"{fname}: 가속도 크기의 중앙값이 {float(g.median()):.2f} g입니다 (1 g 근처여야 함). 단위나 센서 범위 설정을 확인하세요.")
    for k in POWER_COLUMNS:
        if k in df.columns:
            v = pd.to_numeric(df[k], errors="coerce")
            if v.notna().any() and float(v.abs().max()) == 0.0:
                df = df.drop(columns=k)
                data.warn(f"{fname}: {k} 값이 모두 0이라 INA219를 읽지 못한 것으로 보고 쓰지 않습니다 (배선·주소 0x40 확인).")
                continue
            # 이름은 current·voltage인데 값이 mA·mV인 경우 (INA219 최대 3.2 A, 2S 배터리 최대 8.4 V라 이보다 훨씬 크면 단위가 다름)
            med = float(v.abs().median()) if v.notna().any() else 0.0
            if (k == "current" and med > 20) or (k == "voltage" and med > 100):
                df[k] = v / 1000
                data.warn(f"{fname}: {k} 값이 커서({med:g}) {'mA' if k == 'current' else 'mV'}로 보고 1000으로 나눴습니다.")
    unused = [c for c in df.columns if c not in USED_COLUMNS and c not in C.VIB_CHANNELS
              and not any(c == col for cands in UNIT_COLUMNS.values() for col, _ in cands)]
    if unused:
        data.warn(f"쓰지 않은 열: {', '.join(unused)} (열 이름을 알아듣지 못함. 필요하면 config.COLUMN_ALIASES에 추가)")
    return df


class DataError(ValueError):
    """사용자가 고칠 수 있는 데이터 문제. 메시지를 그대로 보여 준다."""


class Dataset:
    """읽은 결과 묶음. robots: 로봇 이름 → 1초 창 목록, fs: 로봇 이름 → 진동 샘플링, warnings: 확인할 점"""

    def __init__(self):
        self.robots, self.warnings, self.fs = {}, [], {}

    def warn(self, text):
        """경고 문구를 모아 둔다. 같은 문구는 한 번만 (파일이 여러 개여도 같은 경고가 반복되지 않게)"""
        if text not in self.warnings:
            self.warnings.append(text)

    def label_of(self, name):
        """로봇의 대표 라벨: 고장 라벨이 하나라도 있으면 그 종류, 라벨이 정상뿐이면 normal, 라벨이 없으면 None"""
        recs = self.robots[name]
        # 고장 라벨이 붙은 창들의 '종류|바퀴' (예: 'bearing|fr') 목록 → 가장 많이 나온 것이 대표 라벨
        kinds = [f"{r['true_kind']}|{r['true_side']}" for r in recs if r["labeled"] and r["true_sev"] > 0]
        if kinds:
            return max(set(kinds), key=kinds.count)
        return "normal|" if any(r["labeled"] for r in recs) else None

    def summary(self):
        """check 명령에서 보이는 로봇별 표: 기록 길이(분), 1초 창 수, 진동 샘플링, 구간 수, 라벨"""
        rows = []
        for name, recs in self.robots.items():
            lbl = self.label_of(name)
            kind = "없음" if lbl is None else C.KIND_KO.get(lbl.split("|")[0], lbl.split("|")[0])
            has = lambda k: "있음" if any(not np.isnan(r.get(k, np.nan)) for r in recs) else "없음"   # 값이 하나라도 있나
            n_enc = sum(has(k) == "있음" for k in C.ENCODER_CHANNELS)
            rows.append({"로봇": name, "기록(분)": round(C.minutes(recs[-1]["period"] + 1), 1) if recs else 0, "1초 창": len(recs),
                         "진동 샘플링(Hz)": round(self.fs.get(name, 0), 1), "구간": len({r["segment"] for r in recs}),
                         "전류": has("current"), "전압": has("voltage"), "엔코더": f"{n_enc}개" if n_enc else "없음",
                         "모터 명령": has("motor_cmd"), "라벨": kind})
        return pd.DataFrame(rows)


def list_files(path=C.DATA_DIR):
    """읽을 CSV 파일 목록. path가 파일이면 그 파일 하나, 폴더면 그 안의 CSV들을 이름순으로 (하위 폴더는 보지 않음)"""
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
    df.columns = [str(c).strip().lower() for c in df.columns]    # 열 이름의 앞뒤 공백 제거 + 소문자 (Time, TIME → time)
    # 열 이름 끝의 단위 표기를 떼고 값을 표준 단위로: '시간(ms)' → 시간 × 0.001, '전류 [mA]' → 전류 × 0.001, 'gx(raw)' → gx
    notes, strip = [], {}
    for c in df.columns:
        m = re.match(r"^(.+?)\s*[\(\[]\s*(.+?)\s*[\)\]]$", c)
        if not m:
            continue
        base, unit = m.group(1).strip(), m.group(2).strip()
        strip[c] = base
        scale = UNIT_SUFFIX.get(unit)
        if scale is not None and scale != 1.0:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            if unit in ("us", "µs", "μs", "ms"):              # 아두이노 시각이면 2^32 한 바퀴 돌기를 먼저 이어 붙임
                v = _unwrap_counter(v)
            df[c] = v * scale
            notes.append(f"'{c}' 열을 단위 {unit}로 보고 × {scale:g} 해서 읽었습니다.")
    df = df.rename(columns=strip)
    df.attrs["notes"] = notes                              # load()가 '확인할 점'으로 옮김
    rename = {}
    for name, aliases in {**C.COLUMN_ALIASES, **C.CHANNEL_ALIASES}.items():
        if name not in df.columns:                         # 표준 이름이 이미 있으면 그대로 두고
            found = next((a for a in aliases if a in df.columns), None)   # 없으면 별칭 중 파일에 있는 첫 번째를 찾아
            if found:
                rename[found] = name                       # 표준 이름으로 바꾼다 (예: timestamp → time, vib_fl → fl_vib)
    return df.rename(columns=rename)


def _numeric(df, name, data, fname):
    """열 하나를 숫자 배열로. 열이 없거나 숫자가 하나도 없으면 None (값을 지어내지 않고 '없음'으로 돌려줌)"""
    if name not in df.columns:
        return None
    col = pd.to_numeric(df[name], errors="coerce")        # 숫자로 못 바꾸는 칸('abc', '-' 등)은 NaN(빈칸)이 됨
    bad = int((col.isna() & df[name].notna()).sum())       # 원래는 무언가 적혀 있었는데 숫자가 아니었던 칸 수
    if bad:
        data.warn(f"{fname}: {name} 열에 숫자가 아닌 값 {bad}개를 빈칸으로 처리했습니다 (그 샘플은 쓰지 않음).")
    return col.to_numpy(float) if col.notna().any() else None


def _seconds(col, fname):
    """time 열 → 초 단위 숫자 배열. 숫자(초)로 적혀 있거나 '2026-11-03 10:00:00.001' 같은 날짜·시각이면 읽는다"""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.99:                          # 99% 넘게 숫자면 이미 초 단위로 본다
        return num.to_numpy(float)
    dt = pd.to_datetime(col, errors="coerce")
    if dt.notna().mean() > 0.99:                           # 날짜·시각이면 첫 샘플부터 몇 초 지났는지로 바꾼다
        return (dt - dt.iloc[0]).dt.total_seconds().to_numpy(float)
    raise DataError(f"{fname}: time 열을 시간으로 읽지 못했습니다. 초 단위 숫자나 날짜·시각 형식이어야 합니다.")


def _labels(col, names):
    """라벨 열의 글자 → 표준 이름 배열 (빈칸은 '', 공백·대소문자 무시, names에 없는 글자는 그대로 둠)"""
    return col.fillna("").astype(str).str.strip().str.lower().map(lambda s: names.get(s, s)).to_numpy()


def _windows(df, fname, data, fs_arg):
    """로봇 1대의 표 → 1초 창 목록 (판정 주기 번호 = 기록 시작부터 몇 번째 PERIOD_S초인가)

    순서: ① MPU6050 6축 열 확인(하나라도 없으면 오류) ② time으로 샘플링 주파수 계산
          ③ 선택 열(전류·전압·구간·문턱·라벨) 준비 ④ time을 1초 단위로 잘라 창마다 진동 파형과 전류·전압 평균을 담음
          ⑤ 진동 샘플이 모자란 창은 버림"""
    df = _standardize(df, fname, data)                    # 스케치마다 다른 단위(µs, 원래 정수값, mV·mA)를 표준으로
    # ① 진동: MPU6050의 6축(가속도 3 + 각속도 3)이 모두 있어야 한다. 빈 축을 다른 값으로 채우지 않는다
    vib = {ch: _numeric(df, ch, data, fname) for ch in C.VIB_CHANNELS}
    missing = [ch for ch in C.VIB_CHANNELS if vib[ch] is None]
    if missing:
        raise DataError(f"{fname}: {', '.join(f'{ch}({C.CHANNEL_KO.get(ch, ch)})' for ch in missing)} 열이 없습니다. "
                        f"MPU6050의 {', '.join(C.VIB_CHANNELS)} 열이 모두 필요합니다. 파일의 열: {list(df.columns)}")
    base = vib[C.VIB_CHANNELS[0]]                          # 샘플 간격은 첫 번째 축(ax) 값이 있는 줄로 잰다
    # ② 샘플링 주파수 = 1 / (진동 샘플 사이 시간 간격의 중앙값). time 열이 없으면 --fs로 알려 줘야 한다 (임의로 가정하지 않음)
    n = len(df)
    if "time" in df.columns:
        t = _seconds(df["time"], fname)
        if np.isnan(t).any():
            raise DataError(f"{fname}: time 열에 빈칸이 있습니다.")
        t = _unwrap_counter(t) if np.nanmax(np.abs(t)) > 2 ** 31 else t   # time 열에 micros() 값을 그대로 적은 경우
        if (np.diff(t) < 0).any():
            raise DataError(f"{fname}: time이 줄어드는 곳이 {int((np.diff(t) < 0).sum())}군데 있습니다. "
                            "기록이 다시 시작됐다면 파일을 나눠 주세요.")
        steps = np.diff(t[~np.isnan(base)])
        steps = steps[steps > 0]
        if not len(steps):
            raise DataError(f"{fname}: time 값이 모두 같아 샘플링 주파수를 알 수 없습니다.")
        fs = 1.0 / float(np.median(steps))
        for scale, unit in ((1e-3, "밀리초"), (1e-6, "마이크로초")):   # time 열이 초가 아니라 ms·µs로 적힌 경우
            if fs < C.MIN_FS and C.MIN_FS <= fs / scale <= 20000:
                t, fs = t * scale, fs / scale
                data.warn(f"{fname}: time 열이 {unit} 단위로 보여 초로 바꿨습니다 (샘플링 {fs:.0f} Hz).")
                break
    elif fs_arg:                                           # time 열이 없으면 사용자가 알려 준 샘플링으로 시각을 계산
        fs = float(fs_arg)
        t = np.arange(n) / fs                              # 0, 1/fs, 2/fs, ... 초
        data.warn(f"{fname}: time 열이 없어 지정한 샘플링 {fs:g} Hz로 시각을 계산했습니다.")
    else:
        raise DataError(f"{fname}: time 열이 없습니다. time 열을 넣거나, 센서의 실제 샘플링 주파수를 --fs 로 알려 주세요.")
    if fs < C.MIN_FS:
        raise DataError(f"{fname}: 진동 샘플링이 {fs:.1f} Hz로 계산됐습니다 ({C.MIN_FS:g} Hz 이상 필요). "
                        "time 열의 단위가 초가 맞는지 확인하세요.")
    if fs < 300:
        data.warn(f"{fname}: 진동 샘플링 {fs:.0f} Hz는 낮아서 고주파(150 Hz 이상) 특징이 거의 비게 됩니다.")

    # ③ 선택 열. INA219 전류·전압: 없으면 None (전력 감시를 하지 않음). 몇 샘플에 한 번만 값이 있어도 된다
    power = {k: _numeric(df, k, data, fname) for k in POWER_COLUMNS}
    # 엔코더(누적 펄스면 16비트 넘침을 이어 붙임)·모터 명령. 진단 표에 보여 주고, 4개가 다 있으면 바퀴 특징에도 씀
    aux = {k: _numeric(df, k, data, fname) for k in AUX_COLUMNS}
    for k in C.ENCODER_CHANNELS:
        if aux[k] is not None and C.ENCODER_IS_COUNTER:
            ok = ~np.isnan(aux[k])
            aux[k][ok] = _unwrap_counter(aux[k][ok], bits=16) if np.nanmax(np.abs(aux[k])) <= 32768 else aux[k][ok]
    # 엔코더 4개가 모두 있으면 바퀴 특징(features.enc_features)을 낼 수 있게 누적 펄스 표(샘플 × 바퀴 4)를 만든다
    enc = None
    if all(aux[k] is not None for k in C.ENCODER_CHANNELS):
        enc = np.column_stack([aux[k] for k in C.ENCODER_CHANNELS])
        if not C.ENCODER_IS_COUNTER:                      # 속도로 적은 경우: 시간에 따라 더해 누적 펄스로 바꿈
            dt = np.r_[0, np.diff(t)]
            enc = np.cumsum(np.nan_to_num(enc) * dt[:, None], axis=0)
    if all(v is None for v in power.values()):
        data.warn(f"{fname}: current·voltage 열이 없어 전류·배터리 감시는 하지 않습니다 (진동만으로 진단).")
    # 구간이 없으면 전체를 한 구간으로 본다 (값을 만들어 넣는 게 아니라 구간 구분을 하지 않는 것)
    seg =df["segment"].fillna(C.DEFAULT_SEGMENT).astype(str).str.strip().to_numpy() if "segment" in df.columns else None
    if seg is None:                                        # 기록기는 SEGMENT를 비우면 이 열을 만들지 않으므로 흔한 경우 → 파일 이름 없이 한 번만 알림
        data.warn("segment 열이 없는 파일은 전체를 한 구간으로 보고 기준을 만듭니다. 장소별 노면 차이는 걸러지지 않습니다.")
    # 문턱: 1/true/yes 등이면 문턱을 넘는 순간 (판정에서 뺄 창)
    bump = df["bump"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t", "y", "yes"]).to_numpy() if "bump" in df.columns else None
    kinds = _labels(df["fault_kind"], KIND_NAMES) if "fault_kind" in df.columns else None      # 고장 종류 (예: bearing)
    wheels = _labels(df["fault_wheel"], WHEEL_NAMES) if "fault_wheel" in df.columns else None  # 고장 바퀴 (예: fr)
    sevs = _numeric(df, "fault_severity", data, fname)                                        # 고장 정도 0~1
    has_label = kinds is not None or sevs is not None      # 라벨 열이 하나라도 있으면 '라벨이 있는 파일'

    # ④ 창 나누기: 시작 시각 기준 몇 번째 1초인지(idx)가 바뀌는 지점이 창의 경계
    idx = np.floor((t - t[0]) / C.WINDOW_S + 1e-9).astype(int)   # 줄마다 '시작부터 몇 번째 1초인가' (0, 0, ..., 1, 1, ...)
    bounds = np.flatnonzero(np.diff(idx)) + 1                     # 그 번호가 바뀌는 줄 = 다음 창이 시작하는 줄
    need, dropped, recs = C.MIN_WINDOW_FILL * fs * C.WINDOW_S, 0, []   # need: 창 하나에 있어야 할 최소 샘플 수 (1000 Hz면 800개)
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):          # a:b = 이번 1초 창에 속한 줄 범위
        waves = {ch: vib[ch][a:b][~np.isnan(vib[ch][a:b])] for ch in C.VIB_CHANNELS}   # 축별 이 1초의 파형 (빈칸은 빼고)
        if min(len(x) for x in waves.values()) < need:               # ⑤ 기록이 끊겨 샘플이 80% 미만이면 버림
            dropped += 1
            continue
        rec = dict(period=int((t[a] - t[0]) // C.PERIOD_S),          # 판정 주기: 기록 시작부터 몇 번째 PERIOD_S초인가
                   segment=seg[a] if seg is not None else C.DEFAULT_SEGMENT,
                   bump=bool(bump[a:b].any()) if bump is not None else False, vib_fs=fs, t=float(t[a]),
                   labeled=False, true_kind="normal", true_side="", true_sev=0.0)
        for ch in C.VIB_CHANNELS:                                    # 축마다 진동 파형
            rec[ch] = waves[ch].astype(np.float32)
        for k, col in power.items():                                 # 전류·전압: 이 1초에 기록된 값들의 평균 (없으면 nan)
            v = col[a:b][~np.isnan(col[a:b])] if col is not None else ()
            rec[k] = float(np.mean(v)) if len(v) else float("nan")
        for k, col in aux.items():                                   # 바퀴 속도[펄스/초]·모터 명령 (없으면 nan)
            rec[k] = float("nan")
            if col is None:
                continue
            ok = ~np.isnan(col[a:b])
            if ok.sum() >= 2 and k in C.ENCODER_CHANNELS and C.ENCODER_IS_COUNTER:   # 누적 펄스: 늘어난 펄스 ÷ 걸린 시간
                tt, vv = t[a:b][ok], col[a:b][ok]
                rec[k] = float((vv[-1] - vv[0]) / (tt[-1] - tt[0])) if tt[-1] > tt[0] else float("nan")
            elif ok.any():
                rec[k] = float(np.mean(col[a:b][ok]))
        if enc is not None:                                          # 바퀴 특징용: 이 1초의 (창 시작부터 시각, 누적 펄스)
            ok = ~np.isnan(enc[a:b]).any(axis=1)
            if ok.sum() >= 3:
                rec["enc_t"] = (t[a:b][ok] - t[a]).astype(np.float32)
                rec["enc_cnt"] = (enc[a:b][ok] - enc[a:b][ok][0]).astype(np.float32)
        if has_label:                                                # 라벨: 창 안의 마지막 고장 종류와 최대 고장 정도
            kind = next((k for k in (kinds[a:b][::-1] if kinds is not None else ()) if k), "")   # 뒤에서부터 첫 빈칸 아닌 종류
            sv = sevs[a:b][~np.isnan(sevs[a:b])] if sevs is not None else ()
            sev = float(np.clip(np.max(sv), 0, 1)) if len(sv) else None                           # 창 안의 최대 고장 정도 (0~1로 자름)
            if not kind and sev:
                kind = "fault"                             # 고장 정도만 있고 종류가 없는 경우
            if kind or sev is not None:
                # 종류가 정상이면 정도 0, 종류만 있고 정도가 없으면 1(완전 고장)로 본다
                sev = 0.0 if kind in ("", "normal") else (1.0 if sev is None else sev)
                side = "" if wheels is None else next((w for w in wheels[a:b][::-1] if w in C.WHEELS), "")   # 고장 바퀴
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
        for n in df.attrs.get("notes", []):                # 열 이름의 단위 표기로 바꾼 것
            data.warn(f"{f.name}: {n}")
        stem = f.name.split(".")[0]                        # 확장자를 뗀 파일 이름 (robot 열이 없을 때 로봇 이름으로 씀)
        # robot 열이 있으면 그 값마다 따로 나눠 처리하고, 없으면 파일 전체가 로봇 1대
        groups = df.groupby(df["robot"].astype(str), sort=False) if "robot" in df.columns else [(stem, df)]
        for robot, part in groups:
            recs, rate = _windows(part, f.name, data, fs)
            have = data.robots.setdefault(str(robot), [])  # 이 로봇에 대해 앞 파일에서 이미 읽은 창들
            offset = have[-1]["period"] + 1 if have else 0    # 같은 로봇의 다음 파일은 이어지는 주기 번호로 붙임
            # 창이 하나도 없는 주기(꺼져 있던 시간)는 건너뛰고 0, 1, 2, ...로 다시 번호를 매긴다
            order = {v: i for i, v in enumerate(dict.fromkeys(r["period"] for r in recs))}
            # 다음 파일의 시각이 앞 파일보다 앞서면(시각이 0부터 다시 시작한 파일 등) 앞 파일 끝 뒤로 밀어 준다
            shift = (have[-1]["t"] + C.WINDOW_S - recs[0]["t"]) if (have and recs and recs[0]["t"] <= have[-1]["t"]) else 0.0
            for r in recs:
                r["period"] = offset + order[r["period"]]
                r["t"] += shift                            # 시각이 앞 파일보다 뒤로 가지 않게
            have += recs
            data.fs[str(robot)] = rate
    return data


def reference_reasons(data):
    """처음 BASELINE_PERIODS주기(5분)로 자기 기준을 만들 수 없는 로봇과 그 이유 → ['짧은기록 (기록이 짧음)', ...]"""
    out = []
    for name, recs in data.robots.items():
        if not recs or recs[-1]["period"] < C.BASELINE_PERIODS:
            out.append(f"{name} (기록이 {C.minutes(C.BASELINE_PERIODS + 1):g}분보다 짧음)")
        elif any(r["true_sev"] > 0 and r["period"] < C.BASELINE_PERIODS for r in recs):
            out.append(f"{name} (처음 {C.minutes(C.BASELINE_PERIODS):g}분에 고장 라벨)")
    return out


def choose_baseline(data, mode=None):
    """정상 기준 방식 결정. auto면 처음 BASELINE_PERIODS주기에 고장 라벨이 있거나 기록이 그보다 짧은 로봇이 있을 때 reference"""
    mode = mode or C.BASELINE_MODE
    if mode != "auto":                                     # 사용자가 --baseline으로 정했으면 그대로
        return mode
    # 처음 5분으로 기준을 못 만드는 로봇(기록이 짧거나 처음부터 고장)이 하나라도 있으면 공통 기준이 필요
    return "reference" if reference_reasons(data) else "per_unit"
