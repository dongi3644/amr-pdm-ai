"""[역할] 실시간 알림: 아두이노(USB) 또는 기록 CSV 재생 → 1초 창 → AI 판정 → 노트북 알림 창
  run.py live 가 부른다.  python run.py live                 아두이노 포트를 찾아 연결 (기록도 실시간_기록/에 저장)
                         python run.py live --replay 파일.csv  기록 파일을 실시간처럼 다시 보냄 (시연·시험)
                         python run.py live --demo           가상 고장 로봇을 실시간처럼 보냄 (로봇 없이 시연)

[흐름] 줄 공급(시리얼 / 파일 / 가상) → Windower: 머리줄을 찾고 1초씩 묶어 dataio._windows로 창(rec) 하나 만들기
       → LiveDiagnoser: 학습한 모델(infer.Predictor)과 전류 감시(power.PowerMonitor)에 넣기
       → 화면: AlertWindow(tkinter 창) 또는 콘솔 출력
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
from pipeline.infer import risk_pct, stage, where_text
from pipeline.power import PowerMonitor

# 시간 열 이름 → 초로 바꾸는 배수 (아두이노 micros()·millis()는 2^32에서 0으로 돌아감)
TIME_COLS = {"t_us": 1e-6, "time_us": 1e-6, "micros": 1e-6, "t_ms": 1e-3, "time_ms": 1e-3, "millis": 1e-3,
             "time": 1.0, "t": 1.0, "time_s": 1.0, "sec": 1.0, "timestamp": 1.0, "시간": 1.0}
SAVE_DIR = C.ROOT / "실시간_기록"                       # 아두이노에서 받은 줄을 그대로 저장하는 곳 (GitHub에 안 올라감)
COLORS = {"정상": ("#2e7d32", "white"), "주의": ("#f9a825", "black"), "경고": ("#ef6c00", "white"),
          "정지 위험": ("#c62828", "white"), "배우는 중": ("#546e7a", "white"), "오류": ("#4a148c", "white"),
          "연결 중": ("#37474f", "white")}


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
    spec.update(encoders=True, glitch=0)
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
        self.n_periods, self.prev_stage = 0, "정상"
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
        """창 하나 → (상태 dict, 새 알림 [(수준, 문구)]). 수준: 0 정보, 1 주의 쪽, 2 경고 쪽"""
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
        for p, s in mon.events[self.n_events:]:            # 상태가 바뀐 때 (2분 연속 규칙, 9점 이상이면 바로)
            name, situation, todo = stage(s, score_at.get(p))
            spot = where_text(mon.where()) if s >= 1 else ""   # 고장 쪽으로 바뀌면: 어디가 평소와 가장 다른가
            alerts.append((min(s, 2), f"{C.minutes(p + 1):g}분: {name} (위험도 {risk_pct(score_at.get(p)) or 0:.0f}%) - {todo}"
                           + (f" · 의심: {spot}" if spot else "")))
            self.prev_stage = name
        self.n_events = len(mon.events)
        if len(mon.period_scores) != self.n_periods:       # 1분 판정이 새로 끝났을 때: 경고 안에서 '정지 위험'(100%)으로 올라가면 알림
            self.n_periods = len(mon.period_scores)
            p, sc = mon.period_scores[-1]
            name = stage(mon.status, sc)[0]
            if name == "정지 위험" and self.prev_stage != "정지 위험":
                spot = where_text(mon.where())
                alerts.append((2, f"{C.minutes(p + 1):g}분: 정지 위험 (위험도 100%) - {C.RISK_TEXT['정지 위험'][1]}"
                               + (f" · 의심: {spot}" if spot else "")))
            self.prev_stage = name
        for p, e in self.power.events[self.n_power:]:      # 전류 높음·배터리 낮음
            alerts.append((1, f"{C.minutes(p + 1):g}분: {e}"))
        self.n_power = len(self.power.events)
        elapsed = rec["t"] + C.WINDOW_S
        last = mon.period_scores[-1][1] if mon.period_scores else None
        name, situation, todo = stage(mon.status, last)
        state = dict(elapsed=elapsed, phase=out["phase"], stage=name, situation=situation, todo=todo,
                     risk_now=risk_pct(out.get("score")), risk_min=risk_pct(last), cause=mon.current_cause(),
                     forecast=None, voltage=rec.get("voltage"), current=rec.get("current"),
                     top=[n for n, _ in out.get("top_feats", [])], calibrated=mon.calibrated,
                     where=[{k: (round(v, 1) if k == "z" else v) for k, v in w.items()} for w in mon.where()])   # 어디가 문제인가 (최근 30초)
        if out["phase"] == "learning":                     # 처음 5분: 이 로봇의 평소 상태를 배우는 중
            state.update(stage="배우는 중", learn=(elapsed, C.minutes(C.BASELINE_PERIODS) * 60))
        if mon.forecast:
            state["forecast"] = (C.minutes(mon.forecast["periods_lo"]), C.minutes(mon.forecast["periods"]))
        return state, alerts


# ---------------------------------------------------------------- 화면: 알림 창
class AlertWindow:
    """tkinter 알림 창. 상태 색 띠 · 위험도 막대 · 상황·할 일 · 원인 · 남은 시간 · 전압·전류 · 알림 목록
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
        for key, label in [("situation", "지금 상황"), ("todo", "할 일"), ("cause", "어디가 문제인가"),
                           ("forecast", "정지 위험까지"), ("power", "전압 · 전류"), ("source", "데이터")]:
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
            col = COLORS["정상"][0] if pct < 33 else COLORS["주의"][0] if pct < 67 else COLORS["경고"][0] if pct < 100 else COLORS["정지 위험"][0]
            c.create_rectangle(0, m, w * min(pct, 100) / 100, h - m, fill=col, outline="")
        for p in (33, 67):
            c.create_line(w * p / 100, 4, w * p / 100, h - 4, fill="#78909c", dash=(3, 2))

    def add_alert(self, level, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.insert(0, f"[{stamp}] {text}")
        self.log.itemconfig(0, fg={0: "#37474f", 1: "#e65100", 2: "#b71c1c"}.get(level, "#37474f"))
        if self.log.size() > 200:
            self.log.delete(200, "end")
        if level >= 1:                                     # 주의·경고 쪽 알림이면 창을 앞으로 + 색 띠 깜빡임
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
        self.lines["cause"].configure(text=st["cause"] or (chr(10).join(f"{w['name']} {abs(w['z']):.1f}배 → {w['hint']}" for w in spots[:2]) or "-"))
        fc = st.get("forecast")
        self.lines["forecast"].configure(text="이미 도달" if st["stage"] == "정지 위험"
                                         else f"약 {fc[1]:.1f}분 (빠르면 {fc[0]:.1f}분)" if fc else "-")
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
    """화면 맨 위 큰 글자: 정상 / 고장 / 고장 의심 / 배우는 중 / 오류 → (글자, 색 이름)
    고장 = 2분 연속 규칙을 거친 단계가 주의 이상. 고장 의심 = 단계는 아직 정상이지만 지금 1초 위험도가 33% 이상 (확인 중)"""
    if st.get("stage") == "오류":
        return "오류", "오류"
    if st.get("stage") == "배우는 중":
        return "배우는 중", "배우는 중"
    if st.get("stage") in ("주의", "경고", "정지 위험"):
        return "고장", st["stage"]
    if (st.get("risk_now") or 0) >= 100 * C.THRESH["caution"] / C.THRESH["danger"]:
        return "고장 의심", "주의"
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
<title>AMR 실시간 고장 알림</title>
<style>
:root{--ink:#1c2630;--muted:#5b6876;--line:#d8dee4;--paper:#fff;--bg:#f3f5f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
#band{padding:22px 24px 18px;color:#fff;transition:background .3s}
#band .top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
#verdict{font-size:56px;font-weight:800;letter-spacing:-.02em;line-height:1.1}
#sub{font-size:18px;margin-top:6px;opacity:.95}
#src{font-size:13px;opacity:.85}
#band.flash{animation:flash .5s step-end 6}@keyframes flash{50%{filter:brightness(1.6)}}
main{max-width:1000px;margin:0 auto;padding:18px 16px 28px;display:grid;gap:16px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
h2{font-size:14px;margin:0 0 10px;color:var(--muted);font-weight:700}
#bar{position:relative;height:26px;background:#e7ebef;border-radius:6px;overflow:hidden}
#fill{height:100%;width:0;transition:width .3s,background .3s}
.mark{position:absolute;top:0;bottom:0;border-left:2px dashed #7b8794}
.scale{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
dl{margin:0;display:grid;grid-template-columns:110px 1fr;gap:8px 12px;font-size:15px}dt{color:var(--muted);font-weight:700}dd{margin:0}
svg{width:100%;height:180px;display:block}
#alerts{list-style:none;margin:0;padding:0;max-height:320px;overflow:auto;font-size:14px}
#alerts li{padding:7px 4px;border-bottom:1px solid var(--line);display:flex;gap:10px}
#alerts .t{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
#alerts .l1{color:#c25e00;font-weight:700}#alerts .l2{color:#b3261e;font-weight:700}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid rgba(255,255,255,.7);background:rgba(255,255,255,.15);color:#fff;cursor:pointer}
.note{font-size:12px;color:var(--muted);margin-top:8px}
.spot{padding:8px 0;border-bottom:1px solid var(--line)}.spot:last-child{border-bottom:0}
.spot .row{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}.spot b{font-size:15px}.spot .z{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px;white-space:nowrap}
.spot .zb{height:6px;background:#e7ebef;border-radius:3px;margin:6px 0}.spot .zb i{display:block;height:100%;border-radius:3px}
.spot .h{font-size:14px}#whereCard.hot{border-color:#ef6c00;box-shadow:0 0 0 2px rgba(239,108,0,.25)}
</style></head><body>
<div id="band"><div class="top"><div><div id="verdict">연결 중</div><div id="sub">데이터를 기다리는 중</div></div>
<div style="text-align:right"><div id="src"></div><div style="margin-top:8px"><button onclick="stopIt()">끝내기</button></div></div></div></div>
<main>
<section class="card"><h2>고장 위험도</h2>
<div id="bar"><div id="fill"></div><div class="mark" style="left:33.3%"></div><div class="mark" style="left:66.7%"></div></div>
<div class="scale"><span>0% 정상</span><span>33% 주의</span><span>67% 경고</span><span>100% 정지 위험</span></div></section>
<div class="grid">
<section class="card"><h2>지금</h2><dl>
<dt>지금 상황</dt><dd id="situation">-</dd><dt>할 일</dt><dd id="todo">-</dd><dt>추정 원인</dt><dd id="cause">-</dd>
<dt>정지 위험까지</dt><dd id="forecast">-</dd><dt>전압 · 전류</dt><dd id="power">-</dd><dt>기록 시간</dt><dd id="elapsed">-</dd></dl>
<div class="note" id="note"></div></section>
<section class="card" id="whereCard"><h2>어디가 문제인가 (최근 30초, 평소와 가장 다른 곳)</h2><div id="where">-</div>
<div class="note">축 방향은 센서를 붙인 방향 기준입니다 (X 화살표 = 로봇 앞쪽). 진동 센서 하나로는 어느 바퀴인지까지는 모르고, 엔코더 4개나 고장 학습 데이터가 있으면 바퀴·원인까지 나옵니다.</div></section>
<section class="card"><h2>위험도 변화</h2><svg id="chart" viewBox="0 0 600 180" preserveAspectRatio="none"></svg></section>
</div>
<section class="card"><h2>알림 (최근이 위)</h2><ul id="alerts"></ul></section>
</main>
<script>
let lastSeq=-1,lastVerdict="",colors={};
const $=id=>document.getElementById(id);
function stopIt(){fetch("/stop",{method:"POST"});$("sub").textContent="끝냈습니다. 이 탭을 닫아도 됩니다.";}
function fmtTime(s){s=Math.floor(s||0);return Math.floor(s/60)+"분 "+String(s%60).padStart(2,"0")+"초";}
function riskColor(p){return p<33.3?colors["정상"]:p<66.7?colors["주의"]:p<100?colors["경고"]:colors["정지 위험"];}
function draw(h){const svg=$("chart");if(!h.length){svg.innerHTML="";return;}
 const t0=h[0][0],t1=Math.max(h[h.length-1][0],t0+60),X=t=>(t-t0)/(t1-t0)*600,Y=v=>170-Math.min(v,100)/100*160;
 let g=`<line x1="0" x2="600" y1="${Y(33.3)}" y2="${Y(33.3)}" stroke="#9aa5b1" stroke-dasharray="4 3"/>`+
 `<line x1="0" x2="600" y1="${Y(66.7)}" y2="${Y(66.7)}" stroke="#9aa5b1" stroke-dasharray="4 3"/>`;
 const pts=h.filter(p=>p[1]!==null).map(p=>X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1));
 if(pts.length){g+=`<polyline fill="none" stroke="#1f6fd1" stroke-width="2" points="${pts.join(" ")}"/>`;
  const last=h.filter(p=>p[1]!==null).slice(-1)[0];g+=`<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="4" fill="${riskColor(last[1])}"/>`;}
 svg.innerHTML=g;}
async function tick(){try{const r=await fetch("/state",{cache:"no-store"});const d=await r.json();colors=d.colors;
 if(d.seq===lastSeq)return;lastSeq=d.seq;$("src").textContent=d.source;
 const [v,c]=d.verdict,st=d.state||{};$("band").style.background=colors[c]||"#37474f";$("verdict").textContent=v;
 if(v!==lastVerdict&&(v==="고장"||v==="고장 의심")){const b=$("band");b.classList.remove("flash");void b.offsetWidth;b.classList.add("flash");}
 lastVerdict=v;
 if(st.stage==="오류"){$("sub").textContent="진행할 수 없음";$("situation").textContent=st.error||"";}
 else if(st.stage==="배우는 중"){const [a,b]=st.learn||[0,300];$("sub").textContent=`이 로봇의 평소 상태를 배우는 중 ${fmtTime(a)} / ${fmtTime(b)}`;
  $("situation").textContent="처음 5분은 판정하지 않고 평소 진동을 모읍니다. 주행·정지·회전을 섞어 주세요.";$("todo").textContent="계속 운행";}
 else if(st.stage){const r=st.risk_now||0;$("sub").textContent=`${st.stage} 단계 · 고장 위험도 ${r.toFixed(0)}%`;
  $("fill").style.width=Math.min(r,100)+"%";$("fill").style.background=riskColor(r);$("situation").textContent=st.situation||"-";$("todo").textContent=st.todo||"-";}
 $("cause").textContent=st.cause||"-";
 $("forecast").textContent=st.stage==="정지 위험"?"이미 도달":(st.forecast?`약 ${st.forecast[1].toFixed(1)}분 (빠르면 ${st.forecast[0].toFixed(1)}분)`:"-");
 const pw=[];if(st.voltage!=null)pw.push(st.voltage.toFixed(2)+" V");if(st.current!=null)pw.push(st.current.toFixed(2)+" A");$("power").textContent=pw.join(" · ")||"-";
 $("elapsed").textContent=st.elapsed!=null?fmtTime(st.elapsed):"-";
 $("note").textContent=st.calibrated===false?"정상 데이터만으로 학습한 모델: '평소와 다르다(고장)'는 확실하지만, 67%·100% 눈금과 원인·남은 시간은 고장 데이터가 있어야 나옵니다.":"";
 const ws=st.where||[],hot=(v==="고장"||v==="고장 의심");$("whereCard").classList.toggle("hot",hot&&ws.length>0);
 $("where").innerHTML=ws.length?ws.map(w=>{const a=Math.abs(w.z),c=a>=6?colors["정지 위험"]:a>=4?colors["경고"]:colors["주의"];
  return `<div class="spot"><div class="row"><b>${w.name}</b><span class="z">${w.what} · ${a.toFixed(1)}배</span></div><div class="zb"><i style="width:${Math.min(a/10,1)*100}%;background:${c}"></i></div><div class="h">→ ${w.hint}</div></div>`;}).join("")
  :`<div class="h" style="color:var(--muted)">평소와 크게 다른 곳 없음 (모든 특징이 평소 범위의 2배 안)</div>`;
 draw(d.history);
 $("alerts").innerHTML=d.alerts.map(a=>`<li class="l${a[1]}"><span class="t">${a[0]}</span><span>${a[2].replace(/[<&]/g,x=>x==="<"?"&lt;":"&amp;")}</span></li>`).join("");
}catch(e){$("sub").textContent="프로그램과 연결이 끊겼습니다 (터미널을 확인하세요)";}}
setInterval(tick,500);tick();
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
