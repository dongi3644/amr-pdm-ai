"""[역할] 실시간 알림: 아두이노(USB) 또는 기록 CSV 재생 → 1초 창 → AI 판정 → 노트북 알림 창
  run.py live 가 부른다.  python run.py live                 아두이노 포트를 찾아 연결 (기록도 실시간_기록/에 저장)
                         python run.py live --replay 파일.csv  기록 파일을 실시간처럼 다시 보냄 (시연·시험)
                         python run.py live --demo           가상 고장 로봇을 실시간처럼 보냄 (로봇 없이 시연)

[흐름] 줄 공급(시리얼 / 파일 / 가상) → Windower: 머리줄을 찾고 1초씩 묶어 dataio._windows로 창(rec) 하나 만들기
       → LiveDiagnoser: 학습한 모델(infer.Predictor)과 전류 감시(power.PowerMonitor)에 넣기
       → 화면: 브라우저 관제 화면(WebView, 기본) · AlertWindow(tkinter 창) · 콘솔 출력
  단계는 정상 · 경미 · 심각 3개 (위험도 0~33 · 33~67 · 67% 이상). 브라우저 화면은 로봇 그림에서 의심 부위를 빨강(심각)·주황(경미 또는
  바퀴를 특정할 수 없는 후보)으로 깜빡이고, 의심 원인 · 의심 부위 · 할 일 · 진동 RMS · 전류 · 전압 · 주행 속도를 보여 준다.
  '#'로 시작하는 줄(아두이노 오류 알림)은 그대로 알림 목록에 띄운다. 소리는 내지 않는다.
  창을 만드는 방식(단위 변환, 각속도가 없으면 엔코더로 gz 계산, 엔코더 종류 판단)은 train·diagnose와 같은 dataio 함수를 쓴다.
"""
import gzip
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from pipeline import dataio as D
from pipeline.features import ENC_FEATS
from pipeline.models import NORMAL_LABEL
from pipeline.infer import risk_pct, stage, suspect_part, where_text
from pipeline.power import PowerMonitor

# 시간 열 이름 → 초로 바꾸는 배수 (아두이노 micros()·millis()는 2^32에서 0으로 돌아감)
TIME_COLS = {"t_us": 1e-6, "time_us": 1e-6, "micros": 1e-6, "t_ms": 1e-3, "time_ms": 1e-3, "millis": 1e-3,
             "time": 1.0, "t": 1.0, "time_s": 1.0, "sec": 1.0, "timestamp": 1.0, "시간": 1.0}
SAVE_DIR = C.ROOT / "실시간_기록"                       # 아두이노에서 받은 줄을 그대로 저장하는 곳 (GitHub에 안 올라감)
COLORS = {"정상": ("#2e7d32", "white"), "경미": ("#f9a825", "#1c2630"), "심각": ("#c62828", "white"),
          "배우는 중": ("#546e7a", "white"), "오류": ("#4a148c", "white"), "연결 중": ("#37474f", "white")}
WIRES = ["wire_fl", "wire_fr", "wire_rl", "wire_rr"]   # 로봇 그림의 모터 배선 (전류 센서가 하나라 어느 선인지는 모름)


def vib_rms(row):
    """그 1초의 차체 진동 크기 [g] = 가속도 축별 RMS를 합친 값 (√(ax² + ay² + az²))"""
    v = [row[f"{a}_rms"] for a in ("ax", "ay", "az") if f"{a}_rms" in row and np.isfinite(row[f"{a}_rms"])]
    return float(np.sqrt(np.sum(np.square(v)))) if v else None


def wheel_speed(rec):
    """그 1초의 주행 속도 [m/s] = 값이 있는 엔코더 바퀴 속도(펄스/초) 크기의 평균 ÷ 1회전 펄스 수 × 바퀴 둘레"""
    v = [abs(rec[k]) for k in C.ENCODER_CHANNELS if k in rec and np.isfinite(rec[k])]
    return float(np.mean(v) / C.ENCODER_PPR * np.pi * C.WHEEL_DIAMETER_M) if v else None


def hot_parts(status, wheel, wiring, battery_low):
    """로봇 그림에서 깜빡일 부위 → {부위: 'red' | 'amber'}
    진동 AI가 경미 이상: 바퀴를 알면 그 바퀴(심각 빨강 · 경미 주황), 모르면 바퀴 4개 모두 주황(후보, 특정 불가)
    배선 접촉 불량 의심: 전류 센서 · 전원 주 배선 · 모터 배선 4개 주황 (INA219가 하나라 어느 선인지는 모름), 배터리 낮음: 배터리 주황"""
    h = {}
    if battery_low:
        h["battery"] = "amber"
    if wiring:
        h.update({"ina": "amber", "harness": "amber", **{w: "amber" for w in WIRES}})
    if status >= 1:
        if wheel:
            h[wheel] = "red" if status >= 2 else "amber"
        else:
            h.update({w: "amber" for w in C.WHEELS})
    return h


# ---------------------------------------------------------------- 줄 공급
def find_port():
    """아두이노로 보이는 시리얼 포트 → (포트 이름 또는 None, 모든 포트 설명 목록)"""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    desc = [f"{p.device} ({p.description})" for p in ports]
    likely = [p.device for p in ports if any(k in (p.description or "") for k in ("Arduino", "CH340", "CH341", "USB Serial", "USB-SERIAL"))]
    if len(likely) == 1:
        return likely[0], desc
    return (ports[0].device if len(ports) == 1 else None), desc


def serial_lines(port, baud, stop):
    """아두이노 시리얼 → 한 줄씩 (글자). 포트를 열면 아두이노가 다시 시작해 '#' 상태 문구와 머리줄부터 보낸다"""
    import serial
    with serial.Serial(port, baud, timeout=1) as s:
        while not stop.is_set():
            raw = s.readline()
            if raw:
                yield raw.decode("utf-8", errors="replace").strip()


def file_lines(path):
    """기록 파일(.csv · .csv.gz) → 한 줄씩"""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            yield line.strip()


def demo_lines(seed=7031):
    """로봇 없이 시연: 가상 로봇 1대(뒤 오른쪽 바퀴 베어링 결함이 서서히 진행)를 아두이노 스케치 출력 모양으로.
    처음 약 6분은 정상 주행이라 그동안 '평소 상태 배우는 중'이 나온다. 코드 점검·시연용 (성능 아님)"""
    import sys
    sys.path.insert(0, str(C.ROOT / "가상테스트"))
    import sim_robots as S                                  # 가상 로봇 생성기 (코드 점검용)
    rng = np.random.default_rng(seed)
    spec = S.complex_spec(rng, "시연_가상로봇", "점진 고장", "bearing", "rr")   # 학습에 안 쓴 seed로 만든 처음 보는 가상 로봇
    spec.update(glitch=0)                                   # 엔코더 4개·모터 명령은 가상 데이터에 항상 들어감
    df = S.make_table(spec)
    t = df["time"].to_numpy()
    raw = {c: np.round(df[c].to_numpy() * (C.RAW_ACCEL_LSB_PER_G if c[0] == "a" else C.RAW_GYRO_LSB_PER_DPS)).astype(int)
           for c in ("ax", "ay", "az", "gx", "gy", "gz")}
    mv = np.round(df["voltage"].ffill().bfill().to_numpy() * 1000).astype(int)
    ma = np.round(df["current"].ffill().bfill().to_numpy() * 1000).astype(int)
    enc = []
    for w in ("fl", "fr", "rl", "rr"):                     # 누적 펄스 → 2 ms마다 센 펄스 (스케치의 encTake)
        c = df[f"enc_{w}"].to_numpy(float)
        ok = ~np.isnan(c)
        cum = np.round(np.interp(t, t[ok], D._unwrap_counter(c[ok], bits=16)))
        enc.append(np.r_[0, np.diff(cum)].astype(int))
    cmd = np.round(df["motor_cmd"].ffill().fillna(0).to_numpy()).astype(int) if "motor_cmd" in df else np.zeros(len(t), int)
    yield (f"# 시연용 가상 로봇 (뒤 오른쪽 바퀴 베어링 결함이 {spec.get('onset', 0):.0f}분쯤부터 서서히 진행, "
           f"기록 {len(t) / 500 / 60:.0f}분) - 성능 아님")
    yield "t_us,ax,ay,az,gx,gy,gz,mV,mA,spd,enc1,enc2,enc3,enc4"
    for i in range(len(t)):
        yield (f"{int(t[i] * 1e6)},{raw['ax'][i]},{raw['ay'][i]},{raw['az'][i]},{raw['gx'][i]},{raw['gy'][i]},{raw['gz'][i]},"
               f"{mv[i]},{ma[i]},{cmd[i]},{enc[0][i]},{enc[1][i]},{enc[2][i]},{enc[3][i]}")


# ---------------------------------------------------------------- 줄 → 1초 창
class Windower:
    """줄을 하나씩 받아 머리줄을 찾고, 1초마다 그 1초의 줄들로 창(rec)을 만든다 → [(종류, 내용), ...]
    종류: 'alert'(아두이노 '#' 알림·상태 문구), 'info'(안내), 'error'(진행 불가), 'rec'(1초 창)"""

    def __init__(self, name):
        self.name, self.header, self.tcol, self.scale = name, None, None, 1.0
        self.rows, self.times, self.idx = [], [], None
        self.t0, self.prev, self.wraps, self.last_raw, self.shift = None, None, 0, None, 0.0
        self.bad, self.seen = 0, set()
        self.known = D._known_names()

    def feed(self, line):
        if not line:
            return []
        if line.startswith("#"):                           # 아두이노 오류·상태 알림
            return [("alert", line.lstrip("#").strip())]
        cells = [c.strip() for c in line.split(",")]
        if self.header is None:                            # 머리줄을 찾기 전: 아는 열 이름이 2개 이상인 줄 = 머리줄
            low = [c.lower() for c in cells]
            if sum(c in self.known for c in low) >= 2:
                tcol = next((c for c in low if c in TIME_COLS), None)
                if tcol is None:
                    return [("error", f"머리줄에 시간 열이 없습니다 (t_us · t_ms · time 중 하나 필요): {line}")]
                self.header, self.tcol, self.scale = low, tcol, TIME_COLS[tcol]
                return [("info", f"머리줄: {line}")]
            return [("alert", line)]                       # 머리줄 앞의 상태 문구 ('shield ok' 등)
        if len(cells) != len(self.header):                 # 칸 수가 다른 깨진 줄
            self.bad += 1
            return []
        try:
            v = float(cells[self.header.index(self.tcol)])
        except ValueError:                                 # 시간 칸이 숫자가 아님: 다시 찍힌 머리줄이거나 상태 문구
            return [] if cells[0].lower() == self.header[0] else [("alert", line)]
        if self.scale < 1:                                 # micros()·millis()가 2^32에서 0으로 돌아가는 것 이어 붙이기
            if self.last_raw is not None and v < self.last_raw - 2 ** 31:
                self.wraps += 1
            self.last_raw = v
            v += self.wraps * 2 ** 32
        sec = v * self.scale + self.shift
        if self.prev is not None and sec < self.prev - 0.5:    # 아두이노가 다시 시작해 시간이 0부터 → 이어지게 밀어 줌
            self.shift += self.prev - sec + 0.002
            sec = self.prev + 0.002
            self.wraps, self.last_raw = 0, None
        self.prev = sec
        if self.t0 is None:
            self.t0 = sec
        t = sec - self.t0
        k = int(t // C.WINDOW_S)
        out = self._flush() if (self.idx is not None and k != self.idx) else []
        self.idx = k
        self.rows.append(cells)
        self.times.append(t)
        return out

    def _flush(self):
        """모아 둔 1초의 줄 → 창 하나 (dataio._windows: train·diagnose와 같은 방식)"""
        if not self.rows:
            return []
        df = pd.DataFrame(self.rows, columns=self.header).drop(columns=[self.tcol])
        df.insert(0, "time", self.times)
        idx = self.idx
        self.rows, self.times = [], []
        df = D.normalize_columns(df)
        junk = D.Dataset()                                 # 이 1초에서 나온 안내 (같은 문구는 한 번만 보여 줌)
        try:
            recs, fs = D._windows(df, self.name, junk, None, enc_check=False)   # 죽은 엔코더 판단은 1초만으로 못 하므로 끔
        except D.DataError as e:
            return [("error", str(e))]
        out = []
        routine = ("샘플이 부족한", "바꿔 읽었습니다", "로 바꿨습니다", "'샘플마다 센 펄스'", "segment 열")   # 늘 나오는 안내는 알림 목록에서 뺌
        for w in junk.warnings:
            key = w.split(":")[-1][:40]
            if key not in self.seen and len(self.seen) < 12 and not any(r in w for r in routine):
                self.seen.add(key)
                out.append(("info", w.replace(f"{self.name}: ", "")))
        for r in recs:
            r["t"] = idx * C.WINDOW_S                      # 기록 시작부터 몇 초째 창인가 (판정 주기 번호를 여기서 다시 매김)
            r["period"] = int(r["t"] // C.PERIOD_S)
            out.append(("rec", r))
        return out

    def finish(self):
        return self._flush()


# ---------------------------------------------------------------- 창 → AI 판정
class LiveDiagnoser:
    """학습한 모델로 창을 하나씩 판정 → 화면에 보여 줄 상태 dict와 새 알림 문구들"""

    def __init__(self, pipe):
        self.pipe = pipe
        self.mon, self.power = pipe.monitor(), PowerMonitor()
        self.n_events = self.n_power = 0
        self.n_periods, self.at100 = 0, False             # 위험도 100%(9점)에 닿아 있는가 (처음 닿을 때 한 번 알림)
        self.checked = False
        self.last_rec = None

    def _check(self, rec):
        """모델이 쓰는 특징을 이 기록으로 만들 수 있나 (진단 diagnose와 같은 확인)"""
        feats = self.pipe.bundle.get("feats", [])
        if any(f in ENC_FEATS for f in feats) and "enc_cnt" not in rec:
            return "이 모델은 엔코더 4개로 학습했는데, 들어오는 데이터에는 엔코더 4개가 없습니다. 같은 방식으로 기록한 데이터로 다시 학습하세요."
        need = {f.split("_")[0] for f in feats if f.split("_")[0] in C.VIB_CHANNELS}
        lack = sorted(ch for ch in need if ch not in rec)
        if lack:
            return f"이 모델은 {', '.join(lack)} 축으로 학습했는데, 들어오는 데이터에는 없습니다 (각속도 열 확인)."
        return None

    def update(self, rec):
        """창 하나 → (상태 dict, 새 알림 [(수준, 문구)]). 수준: 0 정보, 1 경미 쪽, 2 심각 쪽"""
        if not self.checked:
            self.checked = True
            err = self._check(rec)
            if err:
                return dict(stage="오류", error=err), [(2, err)]
        self.last_rec = rec
        self.power.update(rec)
        out = self.mon.update(rec)
        alerts = []
        mon = self.mon
        score_at = dict(mon.period_scores)
        cause = mon.current_cause_label()
        where = mon.where()
        n_enc = sum(1 for k in C.ENCODER_CHANNELS if k in rec and np.isfinite(rec[k]))   # 이 1초에 값이 있는 엔코더 수
        wheel, part = suspect_part(cause, where, n_enc)
        for p, s in mon.events[self.n_events:]:            # 상태가 바뀐 때 (CONFIRM_PERIODS번 연속 규칙, 9점 이상이면 바로)
            name, situation, todo = stage(s, score_at.get(p), cause)
            spot = where_text(where) if s >= 1 else ""     # 고장 쪽으로 바뀌면: 어디가 평소와 가장 다른가
            alerts.append((min(s, 2), f"{C.minutes(p + 1):g}분: {name} (위험도 {risk_pct(score_at.get(p)) or 0:.0f}%)"
                           + (f" · {part}" if s >= 1 else "") + f" - {todo}" + (f" · 의심: {spot}" if spot else "")))
        self.n_events = len(mon.events)
        if len(mon.period_scores) != self.n_periods:       # 1분 판정이 새로 끝났을 때: 심각 안에서 위험도 100%(9점)에 처음 닿으면 알림
            self.n_periods = len(mon.period_scores)
            p, sc = mon.period_scores[-1]
            hit = mon.status >= 2 and sc >= C.THRESH["danger"]
            if hit and not self.at100:
                alerts.append((2, f"{C.minutes(p + 1):g}분: 심각 (위험도 100%, 학습한 고장 기준) · {part} - 운행을 멈추고 바로 점검"))
            self.at100 = hit
        for p, e in self.power.events[self.n_power:]:      # 전류 높음 · 배터리 낮음 · 배선 접촉 불량 의심
            alerts.append((1, f"{C.minutes(p + 1):g}분: {e}" + (f" - {C.FAULT_TODO['wiring']}" if e.startswith("배선 접촉") else "")))
        self.n_power = len(self.power.events)
        elapsed = rec["t"] + C.WINDOW_S
        last = mon.period_scores[-1][1] if mon.period_scores else None
        name, situation, todo = stage(mon.status, last, cause)
        wiring = self.power.wiring == 1
        battery_low = any(e == "배터리 낮음" for _, e in self.power.events)
        # 의심 원인: 원인 분류가 낸 고장 종류 (바퀴는 '의심 부위' 칸이 따로 보여 줌).
        # 원인 모델이 없거나(정상 데이터만으로 학습) 아직 모르면 평소와 가장 다른 곳의 '의심' 문구
        kind = cause.split("|")[0] if cause and cause != NORMAL_LABEL else None
        cause_text = (C.KIND_KO.get(kind, kind) if kind else "원인 미확정" if cause == NORMAL_LABEL else "")             or (where[0]["hint"].split("→")[-1].strip() if mon.status >= 1 and where else "")
        if wiring:                                         # 배선은 전류 규칙이 찾은 것 (진동 AI와 따로)
            cause_text = " / ".join(x for x in [cause_text, C.KIND_KO["wiring"]] if x)
            if mon.status == 0:
                todo = C.FAULT_TODO["wiring"]
        part_text = " / ".join(x for x in [part if mon.status >= 1 else "", "전원·모터 배선 (어느 선인지 특정 불가)" if wiring else "",
                                           "배터리 전압 낮음" if battery_low else ""] if x)
        state = dict(elapsed=elapsed, phase=out["phase"], stage=name, situation=situation, todo=todo,
                     risk_now=risk_pct(out.get("score")), risk_min=risk_pct(last), cause=cause_text,
                     part=part_text, parts=hot_parts(mon.status, wheel, wiring, battery_low), enc_n=n_enc,
                     wiring=wiring, battery_low=battery_low, current_high=self.power.state == 1,
                     vib=vib_rms(out["row"]), speed=wheel_speed(rec),
                     voltage=rec.get("voltage"), current=rec.get("current"),
                     top=[n for n, _ in out.get("top_feats", [])], calibrated=mon.calibrated,
                     where=[{k: (round(v, 1) if k == "z" else v) for k, v in w.items()} for w in where])   # 어디가 문제인가 (최근 30초)
        if out["phase"] == "learning":                     # 처음 5분: 이 로봇의 평소 상태를 배우는 중
            state.update(stage="배우는 중", learn=(elapsed, C.minutes(C.BASELINE_PERIODS) * 60))
        return state, alerts


# ---------------------------------------------------------------- 화면: 알림 창
class AlertWindow:
    """tkinter 알림 창. 상태 색 띠 · 위험도 막대 · 상황·할 일 · 의심 원인 · 의심 부위 · 전압·전류 · 알림 목록
    단계가 올라가면 창을 맨 앞으로 가져오고 색 띠를 3초 동안 깜빡인다 (소리는 내지 않음)"""

    def __init__(self, title, on_close):
        import tkinter as tk
        try:                                               # 윈도우 화면 배율(125%·150%)에서 글자가 흐리게 보이지 않게
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.tk = tk
        self.root = tk.Tk()
        self.sc = max(1.0, self.root.winfo_fpixels("1i") / 96)   # 화면 배율 (창 크기·줄바꿈 폭을 배율에 맞춤)
        self.root.title(f"AMR 실시간 고장 알림 · {title}")
        self.root.geometry(f"{int(560 * self.sc)}x{int(660 * self.sc)}")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        f = "Malgun Gothic"
        self.band = tk.Frame(self.root, height=110)
        self.band.pack(fill="x")
        self.big = tk.Label(self.band, font=(f, 30, "bold"))
        self.big.pack(pady=(12, 0))
        self.sub = tk.Label(self.band, font=(f, 12))
        self.sub.pack(pady=(0, 10))
        self.bar = tk.Canvas(self.root, height=int(34 * self.sc), highlightthickness=0, bg="white")
        self.bar.pack(fill="x", padx=16, pady=(12, 4))
        body = tk.Frame(self.root)
        body.pack(fill="x", padx=16)
        self.lines = {}
        for key, label in [("situation", "지금 상황"), ("todo", "할 일"), ("cause", "의심 원인"), ("part", "의심 부위"),
                           ("power", "전압 · 전류"), ("source", "데이터")]:
            row = tk.Frame(body)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=13, anchor="w", font=(f, 10, "bold"), fg="#455a64").pack(side="left")
            v = tk.Label(row, text="-", anchor="w", justify="left", wraplength=int(420 * self.sc), font=(f, 10))
            v.pack(side="left", fill="x")
            self.lines[key] = v
        tk.Label(self.root, text="알림 (최근이 위)", anchor="w", font=(f, 10, "bold"), fg="#455a64").pack(fill="x", padx=16, pady=(10, 0))
        self.log = tk.Listbox(self.root, font=(f, 10), height=12, activestyle="none")
        self.log.pack(fill="both", expand=True, padx=16, pady=(2, 14))
        self.level, self.flash_until, self.color = -1, 0.0, COLORS["연결 중"]
        self.set_stage("연결 중", "데이터를 기다리는 중")

    def set_stage(self, name, sub):
        bg, fg = COLORS.get(name, COLORS["정상"])
        self.color = (bg, fg)
        for w in (self.band, self.big, self.sub):
            w.configure(bg=bg)
        self.big.configure(text=name, fg=fg)
        self.sub.configure(text=sub, fg=fg)

    def draw_bar(self, pct):
        c = self.bar
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = int(34 * self.sc)
        m = int(8 * self.sc)
        c.create_rectangle(0, m, w, h - m, fill="#eceff1", outline="")
        if pct is not None:
            col = COLORS["정상"][0] if pct < 33.3 else COLORS["경미"][0] if pct < 66.7 else COLORS["심각"][0]
            c.create_rectangle(0, m, w * min(pct, 100) / 100, h - m, fill=col, outline="")
        for p in (33, 67):
            c.create_line(w * p / 100, 4, w * p / 100, h - 4, fill="#78909c", dash=(3, 2))

    def add_alert(self, level, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert(0, f"[{stamp}] {text}")
        self.log.itemconfig(0, fg={0: "#37474f", 1: "#e65100", 2: "#b71c1c"}.get(level, "#37474f"))
        if self.log.size() > 200:
            self.log.delete(200, "end")
        if level >= 1:                                     # 경미·심각 쪽 알림이면 창을 앞으로 + 색 띠 깜빡임
            self.root.deiconify()
            self.root.lift()
            self.flash_until = time.time() + 3.0

    def show(self, st, source):
        if st.get("stage") == "오류":
            self.set_stage("오류", "진행할 수 없음")
            self.lines["situation"].configure(text=st.get("error", ""))
            return
        if st["stage"] == "배우는 중":
            done, total = st["learn"]
            self.set_stage("배우는 중", f"이 로봇의 평소 상태를 배우는 중 {int(done // 60)}:{int(done % 60):02d} / {int(total // 60)}:00")
            self.draw_bar(None)
            self.lines["situation"].configure(text="처음 5분은 판정하지 않고 평소 진동을 모읍니다. 실제 운행처럼 주행·정지·회전을 섞어 주세요.")
            self.lines["todo"].configure(text="계속 운행")
        else:
            risk = st["risk_now"]
            self.set_stage(st["stage"], f"고장 위험도 {risk or 0:.0f}%")
            self.draw_bar(risk)
            self.lines["situation"].configure(text=st["situation"])
            self.lines["todo"].configure(text=st["todo"])
        spots = st.get("where") or []
        self.lines["cause"].configure(text=st.get("cause") or (chr(10).join(f"{w['name']} {abs(w['z']):.1f}배 → {w['hint']}" for w in spots[:2]) or "-"))
        self.lines["part"].configure(text=st.get("part") or "-")
        v, a = st.get("voltage"), st.get("current")
        self.lines["power"].configure(text=" · ".join(x for x in [f"{v:.2f} V" if v is not None and np.isfinite(v) else "",
                                                                  f"{a:.2f} A" if a is not None and np.isfinite(a) else ""] if x) or "-")
        self.lines["source"].configure(text=f"{source} · 기록 {int(st['elapsed'] // 60)}분 {int(st['elapsed'] % 60):02d}초")

    def tick(self):
        """깜빡임: 알림 직후 3초 동안 색 띠를 흰색과 번갈아"""
        if time.time() < self.flash_until and int(time.time() * 4) % 2:
            for w in (self.band, self.big, self.sub):
                w.configure(bg="white")
            self.big.configure(fg=self.color[0])
        else:
            for w in (self.band, self.big, self.sub):
                w.configure(bg=self.color[0])
            self.big.configure(fg=self.color[1])


# ---------------------------------------------------------------- 화면: 브라우저(HTML)
def verdict(st):
    """화면 맨 위 큰 글자: 정상 / 경미 / 심각 / 확인 중 / 배우는 중 / 오류 → (글자, 색 이름)
    경미·심각 = 1분 판정(CONFIRM_PERIODS번 연속 규칙)을 거친 단계. 확인 중 = 단계는 아직 정상이지만 지금 1초 위험도가 33% 이상 (1분 점수가 넘으면 경미)"""
    if st.get("stage") in ("오류", "배우는 중", "경미", "심각"):
        return st["stage"], st["stage"]
    if (st.get("risk_now") or 0) >= 100 * C.THRESH["caution"] / C.THRESH["danger"]:
        return "확인 중", "경미"
    return "정상", "정상"


class WebView:
    """브라우저 알림 화면. 이 컴퓨터 안에서만 열리는 작은 웹 서버(127.0.0.1)가 페이지와 최신 상태(JSON)를 보내고,
    페이지는 0.5초마다 상태를 받아 다시 그린다. 외부 인터넷·라이브러리를 쓰지 않음"""

    def __init__(self, source, port, stop):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.lock = threading.Lock()
        self.data = dict(source=source, state=None, verdict=["연결 중", "연결 중"], alerts=[], history=[], seq=0,
                         colors={k: v[0] for k, v in COLORS.items()})
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):                 # 요청마다 터미널에 찍지 않음
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    with view.lock:
                        body = json.dumps(view.data, ensure_ascii=False, default=float).encode("utf-8")
                    self._send(body, "application/json; charset=utf-8")
                else:
                    self._send(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")

            def do_POST(self):
                if self.path.startswith("/stop"):         # 화면의 '끝내기' 버튼
                    stop.set()
                self._send(b"ok", "text/plain")

        for p in range(port, port + 20):                   # 포트가 쓰이고 있으면 다음 번호
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"알림 화면용 포트 {port}~{port + 19}가 모두 쓰이고 있습니다.")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def push_state(self, st):
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in st.items()}
        with self.lock:
            self.data["state"] = clean
            self.data["verdict"] = list(verdict(st))
            if "elapsed" in st:
                self.data["history"].append([round(st["elapsed"], 1), None if st.get("risk_now") is None else round(st["risk_now"], 1)])
                self.data["history"] = self.data["history"][-1800:]   # 최근 30분
            self.data["seq"] += 1

    def push_alert(self, level, text):
        with self.lock:
            self.data["alerts"].insert(0, [datetime.now().strftime("%H:%M:%S"), int(level), text])
            del self.data["alerts"][200:]
            self.data["seq"] += 1

    def close(self):
        self.server.shutdown()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMR 구동계 실시간 관제</title>
<style>
:root{--ink:#1c2630;--muted:#5b6876;--line:#d8dee4;--paper:#fff;--bg:#f3f5f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;font-size:14px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:12px 20px;background:#1c2630;color:#fff}
header h1{font-size:18px;margin:0;font-weight:800}
.tag{font-size:12px;background:#ffcc80;color:#3e2700;border-radius:4px;padding:3px 8px;font-weight:700}
button{font:inherit;padding:6px 12px;border-radius:7px;border:1px solid rgba(255,255,255,.4);background:rgba(255,255,255,.12);color:#fff;cursor:pointer}
main{max-width:1200px;margin:0 auto;padding:16px 20px 28px;display:grid;gap:14px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
h2{font-size:13px;margin:0 0 10px;color:var(--muted);font-weight:700}
#band{padding:16px 20px;color:#fff;border-radius:10px 10px 0 0;transition:background .3s}
#band.flash{animation:flash .5s step-end 6}@keyframes flash{50%{filter:brightness(1.5)}}
#verdict{font-size:40px;font-weight:800;line-height:1.1}
#sub{font-size:16px;margin-top:4px;opacity:.95}
#bar{position:relative;height:22px;background:#e7ebef;border-radius:6px;overflow:hidden}
#fill{height:100%;width:0;transition:width .3s,background .3s}
.mark{position:absolute;top:0;bottom:0;border-left:2px dashed #7b8794}
.scale{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:4px}
.grid2{display:grid;grid-template-columns:1.3fr 1fr;gap:14px}
@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
dl{margin:0;display:grid;grid-template-columns:100px 1fr;gap:7px 12px}dt{color:var(--muted);font-weight:700}dd{margin:0}
.spot{padding:8px 0;border-bottom:1px solid var(--line)}.spot:last-child{border-bottom:0}
.spot .row{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.spot .z{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.spot .zb{height:6px;background:#e7ebef;border-radius:3px;margin:6px 0}.spot .zb i{display:block;height:100%;border-radius:3px}
.note{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.5}
#alerts{list-style:none;margin:0;padding:0;max-height:260px;overflow:auto;font-size:13px}
#alerts li{padding:6px 2px;border-bottom:1px solid var(--line);display:flex;gap:10px}
#alerts .t{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
#alerts .l1{color:#b26a00;font-weight:700}#alerts .l2{color:#b3261e;font-weight:700}
svg.chart{width:100%;height:150px;display:block}
#robot{width:100%;height:auto;display:block}
#robot .part{cursor:pointer}
#robot .shape{transition:stroke .2s}
#robot .hot-red .shape{stroke:#d32f2f;stroke-width:5;animation:blink 1s infinite}
#robot .hot-amber .shape{stroke:#ef8f00;stroke-width:5;animation:blink 1.4s infinite}
#robot .hot-red .lbl,#robot .hot-amber .lbl{font-weight:800}
#robot .hot-red .lbl{fill:#c62828}#robot .hot-amber .lbl{fill:#b26a00}
#robot .badge{display:none}
#robot .hot-red .badge,#robot .hot-amber .badge{display:block}
#robot .lbl{font-size:13px;fill:#37474f}
@keyframes blink{50%{stroke-opacity:.25}}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);flex-wrap:wrap;margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body>
<header><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap"><h1>AMR 구동계 실시간 관제</h1><span class="tag" id="src">연결 중</span></div>
<div><button onclick="stopIt()">끝내기</button></div></header>
<main>
<div class="card" style="padding:0">
 <div id="band"><div id="verdict">연결 중</div><div id="sub">데이터를 기다리는 중</div></div>
 <div style="padding:14px 16px"><h2>고장 위험도</h2>
 <div id="bar"><div id="fill"></div><div class="mark" style="left:33.3%"></div><div class="mark" style="left:66.7%"></div></div>
 <div class="scale"><span>0% 정상</span><span>33% 경미</span><span>67% 심각</span><span>100% 학습한 고장 기준</span></div></div>
</div>
<div class="grid2">
 <div class="card"><h2>고장 부위</h2>
  <svg id="robot" viewBox="0 80 560 290" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="legend"><span><i style="background:#d32f2f"></i>의심 부위 (심각)</span><span><i style="background:#ef8f00"></i>의심 부위 (경미) · 후보 부위 (특정 불가)</span><span>부위를 누르면 설명이 나옵니다</span></div>
  <div class="note" id="encNote"></div></div>
 <div class="card"><h2>지금</h2><dl>
  <dt>지금 상황</dt><dd id="situation">-</dd>
  <dt>의심 원인</dt><dd id="cause">-</dd>
  <dt>의심 부위</dt><dd id="part">-</dd>
  <dt>할 일</dt><dd id="todo">-</dd>
  <dt>진동 RMS</dt><dd id="vVib">-</dd>
  <dt>전류</dt><dd id="vCur">-</dd>
  <dt>전압</dt><dd id="vVolt">-</dd>
  <dt>주행 속도</dt><dd id="vSpd">-</dd>
  <dt>기록 시간</dt><dd id="elapsed">-</dd></dl>
  <div class="note" id="note"></div>
  <h2 style="margin-top:16px">어디가 평소와 다른가 (최근 30초)</h2><div id="where">-</div>
  <div class="note">축 방향은 센서를 붙인 방향 기준입니다 (X 화살표 = 로봇 앞쪽).</div></div>
</div>
<div class="grid2">
 <div class="card"><h2>위험도 변화 (최근 5분)</h2><svg class="chart" id="chart" viewBox="0 0 600 150" preserveAspectRatio="none"></svg></div>
 <div class="card"><h2>알림 (최근이 위)</h2><ul id="alerts"></ul></div>
</div>
</main>
<script>
let lastSeq=-1,lastVerdict="",colors={},lockedNote=false;
const $=id=>document.getElementById(id);
const TXT={"경미":"#1c2630"};
const PART_KO={fl:"앞 왼쪽 바퀴·모터",fr:"앞 오른쪽 바퀴·모터",rl:"뒤 왼쪽 바퀴·모터",rr:"뒤 오른쪽 바퀴·모터",
 wire_fl:"앞 왼쪽 모터 배선",wire_fr:"앞 오른쪽 모터 배선",wire_rl:"뒤 왼쪽 모터 배선",wire_rr:"뒤 오른쪽 모터 배선",
 harness:"전원 주 배선",battery:"배터리",board:"제어기",imu:"진동 센서",ina:"전류 센서"};
const WHEEL_INFO="바퀴·모터: 불균형·변형, 체결 풀림, 마찰 증가(베어링·걸림)가 생기는 곳";
const WIRE_INFO="모터 배선: 커넥터가 헐거우면 전류가 순간적으로 끊김 (전류 센서가 하나라 어느 선인지는 특정하지 않음)";
const PART_INFO={fl:WHEEL_INFO,fr:WHEEL_INFO,rl:WHEEL_INFO,rr:WHEEL_INFO,wire_fl:WIRE_INFO,wire_fr:WIRE_INFO,wire_rl:WIRE_INFO,wire_rr:WIRE_INFO,
 harness:"전원 주 배선: 배터리 → 전류 센서 → 모터 드라이버",battery:"배터리 7.4 V 2셀: 전압이 7.0 V 아래면 '배터리 낮음' (규칙 기반 감시, AI 아님)",
 board:"제어기(아두이노): 센서 값을 보냄",imu:"진동 센서(MPU6050): 차체 진동 6축",ina:"전류 센서(INA219): 전류·전압, 전류 순간 끊김으로 배선 접촉 불량 감시"};
function stopIt(){fetch("/stop",{method:"POST"});$("sub").textContent="끝냈습니다. 이 탭을 닫아도 됩니다.";}
function fmtTime(s){s=Math.floor(s||0);return Math.floor(s/60)+"분 "+String(s%60).padStart(2,"0")+"초";}
function riskColor(p){return p<33.3?colors["정상"]:p<66.7?colors["경미"]:colors["심각"];}
function esc(t){return String(t).replace(/[<&]/g,x=>x==="<"?"&lt;":"&amp;");}
function buildRobot(){
 const W=(id,cx,cy,rx,ry,lx,ly)=>`<g class="part" data-p="${id}"><ellipse class="shape" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="#263238" stroke="#11181d" stroke-width="2"/>
  <ellipse cx="${cx}" cy="${cy}" rx="${rx*0.45}" ry="${ry*0.45}" fill="#90a4ae"/><line x1="${cx}" y1="${cy-ry*0.45}" x2="${cx}" y2="${cy+ry*0.45}" stroke="#546e7a" stroke-width="3"/>
  <text class="lbl" x="${lx}" y="${ly}" text-anchor="middle">${PART_KO[id].replace("·모터","")}</text>
  <g class="badge"><circle cx="${cx+rx-2}" cy="${cy-ry}" r="11" fill="#d32f2f"/><text x="${cx+rx-2}" y="${cy-ry+5}" text-anchor="middle" fill="#fff" font-weight="800" font-size="14">!</text></g></g>`;
 const P=(id,pts,fill,lx,ly,label)=>`<g class="part" data-p="${id}"><polygon class="shape" points="${pts}" fill="${fill}" stroke="#455a64" stroke-width="1.5"/>
  ${label?`<text class="lbl" x="${lx}" y="${ly}" text-anchor="middle">${label}</text>`:""}
  <g class="badge"><circle cx="${lx}" cy="${ly-22}" r="10" fill="#d32f2f"/><text x="${lx}" y="${ly-17}" text-anchor="middle" fill="#fff" font-weight="800" font-size="13">!</text></g></g>`;
 const Wire=(id,d,color)=>`<g class="part" data-p="${id}"><path class="shape" d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round"/><g class="badge"></g></g>`;
 $("robot").innerHTML=`<ellipse cx="280" cy="300" rx="250" ry="34" fill="#000" opacity=".06"/>
  ${W("rl",104,150,22,36,70,100)}${W("rr",456,150,22,36,490,100)}
  <rect x="124" y="132" width="40" height="18" rx="4" fill="#c9a227" stroke="#8d6e00"/><rect x="396" y="132" width="40" height="18" rx="4" fill="#c9a227" stroke="#8d6e00"/>
  <polygon points="150,110 410,110 470,212 90,212" fill="#dfe4e7" stroke="#78909c" stroke-width="2"/>
  <polygon points="90,212 470,212 470,236 90,236" fill="#b0bec5" stroke="#78909c" stroke-width="2"/>
  <rect x="104" y="214" width="46" height="18" rx="4" fill="#c9a227" stroke="#8d6e00"/><rect x="410" y="214" width="46" height="18" rx="4" fill="#c9a227" stroke="#8d6e00"/>
  ${Wire("wire_rl","M232,178 C200,170 180,150 160,141","#e53935")}${Wire("wire_rr","M330,178 C360,170 380,150 400,141","#e53935")}
  ${Wire("wire_fl","M240,195 C200,205 160,215 128,222","#e53935")}${Wire("wire_fr","M322,195 C360,205 400,215 432,222","#e53935")}
  ${Wire("harness","M280,150 L280,162 M222,138 C205,140 196,146 192,152","#212121")}
  ${P("battery","228,120 332,120 344,148 216,148","#37474f",280,104,"배터리")}
  ${P("ina","178,146 210,146 214,158 174,158","#7e57c2",196,132,"전류 센서")}
  ${P("board","226,160 336,160 350,196 212,196","#1565c0",281,182,"")}
  <text x="281" y="183" text-anchor="middle" fill="#fff" font-size="12" font-weight="700" pointer-events="none">제어기</text>
  ${P("imu","356,166 384,166 388,180 352,180","#00897b",404,200,"진동 센서")}
  ${W("fl",70,234,26,44,48,312)}${W("fr",490,234,26,44,512,312)}
  <text x="280" y="268" text-anchor="middle" font-size="12" fill="#78909c">▲ 앞</text>
  <text id="callout" x="280" y="338" text-anchor="middle" font-size="17" font-weight="800" fill="#2e7d32"></text>
  <text x="280" y="362" text-anchor="middle" font-size="11" fill="#90a4ae">빨간 선 = 모터 배선 · 검은 선 = 전원 주 배선</text>`;
 document.querySelectorAll("#robot .part").forEach(g=>g.addEventListener("click",()=>{const p=g.dataset.p;lockedNote=true;
  $("encNote").innerHTML=`<b>${PART_KO[p]}</b> — ${PART_INFO[p]}`;}));
}
function draw(h){const svg=$("chart");const pts0=h.filter(p=>p[1]!==null);if(!pts0.length){svg.innerHTML="";return;}
 const t1=pts0[pts0.length-1][0],t0=t1-300,X=t=>(t-t0)/300*600,Y=v=>145-Math.min(v,100)/100*140;
 let g=[33.3,66.7].map(v=>`<line x1="0" x2="600" y1="${Y(v)}" y2="${Y(v)}" stroke="#9aa5b1" stroke-dasharray="4 3"/>`).join("");
 const last=pts0[pts0.length-1];
 g+=`<polyline fill="none" stroke="${riskColor(last[1])}" stroke-width="2.5" points="${pts0.filter(p=>p[0]>=t0).map(p=>X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join(" ")}"/>`;
 svg.innerHTML=g;}
function num(v,d,u){return v==null?"-":v.toFixed(d)+" "+u;}
async function tick(){try{const r=await fetch("/state",{cache:"no-store"});const d=await r.json();colors=d.colors;
 if(d.seq===lastSeq)return;lastSeq=d.seq;$("src").textContent=d.source;
 const [v,c]=d.verdict,st=d.state||{};$("band").style.background=colors[c]||"#37474f";$("band").style.color=TXT[c]||"#fff";$("verdict").textContent=v;
 if(v!==lastVerdict&&(v==="경미"||v==="심각"||v==="확인 중")){const b=$("band");b.classList.remove("flash");void b.offsetWidth;b.classList.add("flash");}
 lastVerdict=v;
 const bad=st.stage==="경미"||st.stage==="심각";
 if(st.stage==="오류"){$("sub").textContent="진행할 수 없음";$("situation").textContent=st.error||"";}
 else if(st.stage==="배우는 중"){const [a,b]=st.learn||[0,300];$("sub").textContent=`이 로봇의 평소 상태를 배우는 중 ${fmtTime(a)} / ${fmtTime(b)}`;
  $("situation").textContent="처음 5분은 판정하지 않고 평소 진동을 모읍니다. 주행·정지·회전을 섞어 주세요.";$("todo").textContent="계속 운행";}
 else if(st.stage){const r=st.risk_now||0;
  $("sub").textContent=`고장 위험도 ${r.toFixed(0)}%`+(bad?` — ${st.part||""} ${st.cause?st.cause+" 의심":""}`:v==="확인 중"?" — 평소와 다른 진동, 1분 점수가 넘으면 경미":" — 진동은 평소 범위 안")+(!bad&&st.wiring?" · 배선 접촉 불량 의심 (전류 순간 끊김)":"");
  $("fill").style.width=Math.min(r,100)+"%";$("fill").style.background=riskColor(r);$("situation").textContent=st.situation||"-";$("todo").textContent=st.todo||"-";}
 $("cause").textContent=st.cause||"-";$("part").textContent=st.part||"-";
 $("vVib").textContent=num(st.vib,3,"g");$("vCur").textContent=num(st.current,2,"A")+(st.current_high?" (최근 2분 평소보다 높음)":"");
 $("vVolt").textContent=num(st.voltage,2,"V")+(st.battery_low?" (7.0 V 아래)":"");$("vSpd").textContent=num(st.speed,2,"m/s");
 $("elapsed").textContent=st.elapsed!=null?fmtTime(st.elapsed):"-";
 $("note").textContent=st.calibrated===false?"정상 데이터만으로 학습한 모델: '평소와 다르다'는 확실하지만, 67%·100% 눈금과 원인 분류는 고장 데이터가 있어야 나옵니다.":"";
 const hot=st.parts||{};
 document.querySelectorAll("#robot .part").forEach(g=>{g.classList.remove("hot-red","hot-amber");const h=hot[g.dataset.p];if(h)g.classList.add("hot-"+h);});
 if(!lockedNote)$("encNote").innerHTML=(st.enc_n||0)<4
  ?`이 로봇은 엔코더가 ${st.enc_n||0}개라 <b>바퀴 4개 중 어느 것인지는 특정하지 않습니다</b> (원인 분류가 바퀴를 내지 않으면 후보 4개를 주황으로 표시). 엔코더 4개를 달면 한 바퀴만 다르게 도는 것으로 위치를 찾습니다.`
  :"엔코더 4개: 한 바퀴만 다르게 도는 정도로 고장 바퀴를 찾습니다. 진동 센서만으로는 어느 바퀴인지 알 수 없습니다.";
 const co=$("callout");const anyHot=Object.keys(hot).length>0;
 co.textContent=bad?`⚠ ${st.part||""}${st.cause?" · "+st.cause:""}`:anyHot?`⚠ ${st.part||""}`:"이상 부위 없음";
 co.setAttribute("fill",st.stage==="심각"?"#c62828":(bad||anyHot)?"#b26a00":"#2e7d32");
 const ws=st.where||[];
 $("where").innerHTML=ws.length?ws.map(w=>{const a=Math.abs(w.z),cc=a>=6?"#c62828":a>=3?"#ef8f00":"#90a4ae";
  return `<div class="spot"><div class="row"><b>${esc(w.name)}</b><span class="z">${esc(w.what)} · 평소의 ${a.toFixed(1)}배</span></div><div class="zb"><i style="width:${Math.min(a/10,1)*100}%;background:${cc}"></i></div><div>→ ${esc(w.hint)}</div></div>`;}).join("")
  :`<div class="note">평소 범위를 2배 넘게 벗어난 값이 없습니다.</div>`;
 draw(d.history);
 $("alerts").innerHTML=d.alerts.map(a=>`<li class="l${a[1]}"><span class="t">${a[0]}</span><span>${esc(a[2])}</span></li>`).join("");
}catch(e){$("sub").textContent="프로그램과 연결이 끊겼습니다 (터미널을 확인하세요)";}}
buildRobot();setInterval(tick,500);tick();
</script></body></html>"""


# ---------------------------------------------------------------- 실행
def run(lines, source, pipe, view="html", pace=None, save=None, labels=None, stop=None, max_seconds=None, port=8765, open_browser=True):
    """줄 공급기 lines → 판정 → 화면(view: html 브라우저 · window tkinter 창 · console 터미널). pace: 1초 창마다 기다릴 시간[s] (재생 속도, None이면 기다리지 않음 = 실시간 입력)
    save: 받은 줄을 그대로 저장할 파일, labels: 저장할 때 데이터 줄 끝에 붙일 라벨 {열 이름: 값}
    max_seconds: 이 만큼(기록 시간) 처리하면 멈춤 (시험용)"""
    stop = stop or threading.Event()
    q = queue.Queue()
    name = "실시간"
    win = Windower(name)
    diag = LiveDiagnoser(pipe)
    labels = labels or {}

    def worker():
        f = None
        try:
            if save:
                Path(save).parent.mkdir(parents=True, exist_ok=True)
                f = open(save, "w", encoding="utf-8", newline="\n")
                q.put(("info", f"받은 데이터를 저장합니다: {save}"))
            for line in lines:
                if stop.is_set():
                    break
                if f:                                      # 받은 줄을 그대로 저장 (머리줄·데이터 줄에는 라벨 열을 덧붙임)
                    if win.header is None and labels and line and not line.startswith("#") and \
                            sum(c.strip().lower() in win.known for c in line.split(",")) >= 2:
                        f.write(line + "," + ",".join(labels) + "\n")
                    elif win.header is not None and labels and line and not line.startswith("#") and line.count(",") + 1 == len(win.header):
                        f.write(line + "," + ",".join(str(v) for v in labels.values()) + "\n")
                    else:
                        f.write(line + "\n")
                for kind, item in win.feed(line):
                    if kind == "rec":
                        st, alerts = diag.update(item)
                        for a in alerts:
                            q.put(("alert",) + a)
                        q.put(("state", st))
                        if st.get("stage") == "오류":
                            stop.set()
                            break
                        if pace:
                            time.sleep(pace)
                        if max_seconds and st["elapsed"] >= max_seconds:
                            stop.set()
                    elif kind == "alert":
                        q.put(("alert", 1 if any(k in item for k in ("FAIL", "실패", "없음", "낮음", "이상", "오류", "초과")) else 0,
                               f"아두이노: {item}"))
                    else:
                        q.put((kind, item))
            for kind, item in win.finish():
                if kind == "rec":
                    st, alerts = diag.update(item)
                    for a in alerts:
                        q.put(("alert",) + a)
                    q.put(("state", st))
            q.put(("end", "데이터가 끝났습니다."))
        except Exception as e:                             # 포트가 빠지는 등: 창에 알리고 멈춤
            q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if f:
                f.close()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    summary = {"states": 0, "alerts": [], "last": None}

    def handle(msg, ui):
        kind = msg[0]
        if kind == "state":
            summary["states"] += 1
            summary["last"] = msg[1]
            if ui:
                ui.show(msg[1], source)
            elif msg[1].get("stage") == "오류":
                print("오류:", msg[1].get("error"))
            elif int(msg[1]["elapsed"]) % 60 == 0:      # 콘솔: 1분마다 한 줄
                st = msg[1]
                print(f"[{int(st['elapsed'] // 60):3d}분] {st['stage']}"
                      + ("" if st["stage"] == "배우는 중" else f" · 위험도 {st['risk_now'] or 0:.0f}% · {st['todo']}")
                      + (f" · 추정 원인 {st['cause']}" if st.get("cause") else ""), flush=True)
        elif kind == "alert":
            summary["alerts"].append(msg[2])
            if ui:
                ui.add_alert(msg[1], msg[2])
            else:
                print(("⚠ " if msg[1] >= 1 else "  ") + msg[2], flush=True)
        elif kind in ("info", "error", "end"):
            text = ("오류: " if kind == "error" else "") + msg[1]
            if ui:
                ui.add_alert(2 if kind == "error" else 0, text)
            else:
                print(text, flush=True)
        return kind

    if view == "html":                                  # 브라우저 화면: 주 스레드는 큐를 비우며 화면 상태만 갱신
        web = WebView(source, port, stop)
        print(f"알림 화면: {web.url}  (브라우저가 안 열리면 이 주소를 직접 여세요. 끝내려면 화면의 '끝내기' 또는 Ctrl+C)", flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(web.url)
        try:
            while not stop.is_set():
                try:
                    msg = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                kind = msg[0]
                if kind == "state":
                    summary["states"] += 1
                    summary["last"] = msg[1]
                    web.push_state(msg[1])
                elif kind == "alert":
                    summary["alerts"].append(msg[2])
                    web.push_alert(msg[1], msg[2])
                else:
                    web.push_alert(2 if kind == "error" else 0, ("오류: " if kind == "error" else "") + msg[1])
                    if kind in ("end", "error") and max_seconds:
                        stop.set()
        except KeyboardInterrupt:
            pass
        stop.set()
        web.close()
        return summary

    if view == "console":
        while th.is_alive() or not q.empty():
            try:
                if handle(q.get(timeout=0.2), None) in ("end", "error"):
                    break
            except queue.Empty:
                continue
        stop.set()
        return summary

    ui = AlertWindow(source, lambda: (stop.set(), ui.root.destroy()))

    def poll():
        try:
            for _ in range(200):
                handle(q.get_nowait(), ui)
        except queue.Empty:
            pass
        ui.tick()
        if not stop.is_set() or not q.empty():
            ui.root.after(100, poll)

    ui.root.after(100, poll)
    ui.root.mainloop()
    stop.set()
    return summary


def save_path(robot):
    """아두이노 기록 저장 파일 이름: 실시간_기록/<로봇>_<날짜_시각>.csv"""
    return SAVE_DIR / f"{robot}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
