"""전류·배터리 감시: INA219의 전류[A]·전압[V]으로 '평소보다 전류를 많이 먹는가', '배터리가 얼마나 남았나'를 본다

[역할] 학습 데이터가 없어도 동작하는 감시기. 진단할 때 로봇마다 처음 BASELINE_PERIODS주기(기본 5분)를
       그 로봇의 '평소'로 삼고, 그 뒤의 전류를 평소와 비교한다. 진동 AI(infer.Predictor)와 따로 돌고 따로 표시된다.
[흐름에서 위치] dataio의 1초 창(current, voltage 평균) → (여기) → workflow.diagnose 표의 전류·배터리 칸
[배터리 센서가 없는 대신]
    배터리 전압   INA219의 전압 = 배터리 전압 (INA219를 배터리와 모터 드라이버 사이에 달았을 때)
    사용량[mAh]   전류를 시간에 따라 더함 (1초 창 평균 전류 × 1초). 기록 시작 이후 쓴 양이다
                  (시작할 때 배터리가 얼마나 차 있었는지는 모르므로 '남은 양'이 아니라 '쓴 양'으로 보여 준다)

[판정 규칙] (모두 config.py에서 바꿀 수 있음)
    전류 높음     처음 5분 동안 1초 평균 전류의 99.5% 지점(POWER_QUANTILE)을, 한 주기의 창 절반 넘게 넘는 일이 2주기 연속
                  → 모터가 평소보다 힘들게 돈다는 뜻 (마찰·걸림·베어링 손상·과적 등). 정상으로 돌아올 때도 2주기 연속
    배터리 낮음   한 주기 전압의 중앙값이 BATTERY_LOW_V(7.0 V) 아래 (모터가 돌 때 전압이 잠깐 떨어지는 것은 중앙값으로 걸러짐)
    평소 대비(배) 이번 주기 평균 전류 ÷ 처음 5분 평균 전류 (1.3이면 30% 더 먹음 = 배터리가 그만큼 빨리 닳음)
"""
import numpy as np

import config as C

STATE_KO = ["정상", "높음"]


class PowerMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.base = {}                    # 처음 5분: 구간 이름 → 1초 평균 전류 목록
        self.limit = None                 # 처음 5분이 끝나면: 구간 이름 → 전류 상한 (99.5% 지점), '전체' 키는 모든 구간 합친 값
        self.base_mean = None             # 처음 5분 평균 전류 (평소 대비 배수의 분모)
        self.cur_period, self.buf = None, []   # 지금 주기 번호, 이번 주기 창들의 (전류, 전압, 상한)
        self.used_mah = 0.0               # 기록 시작 이후 사용량 [mAh]
        self.state, self.pending = 0, (0, 0)   # 전류 상태 (0 정상, 1 높음), (목표 상태, 연속 주기 수)
        self.periods = {}                 # 주기 번호 → 그 주기 결과 dict (표에 들어갈 값)
        self.events = []                  # (주기, 문구) 상태 변경 이력. 예: (12, '전류 높음')
        self.has_current = self.has_voltage = False

    # ---------------------------------------------------------------- 창마다
    def update(self, rec):
        """1초 창 하나: 사용량 더하기 → (처음 5분이면) 평소 전류 모으기 → 이번 주기 버퍼에 넣기"""
        if rec["period"] != self.cur_period:              # 주기가 바뀌면 지난 주기를 마무리
            if self.cur_period is not None:
                self._finish_period(self.cur_period)
            self.cur_period = rec["period"]
        cur, volt = rec["current"], rec["voltage"]
        self.has_current |= not np.isnan(cur)
        self.has_voltage |= not np.isnan(volt)
        if not np.isnan(cur):
            self.used_mah += max(cur, 0.0) * C.WINDOW_S / 3.6   # A × s ÷ 3.6 = mAh (음수는 충전·잡음이라 빼지 않음)
        seg = rec["segment"]
        if rec["period"] < C.BASELINE_PERIODS:            # 처음 5분: 평소 전류를 모으기만 함 (문턱 창 제외)
            if not np.isnan(cur) and not rec["bump"]:
                self.base.setdefault(seg, []).append(cur)
        elif self.limit is None and self.base:            # 5분이 지난 첫 창에서 평소 기준을 만든다
            every = np.concatenate([np.asarray(v) for v in self.base.values()])
            self.limit = {s: float(np.quantile(v, C.POWER_QUANTILE)) for s, v in self.base.items() if len(v) >= 30}
            self.limit["__전체__"] = float(np.quantile(every, C.POWER_QUANTILE))   # 창이 적은 구간·처음 보는 구간은 전체 기준
            self.base_mean = float(np.mean(every))
        lim = self.limit.get(seg, self.limit["__전체__"]) if self.limit else None
        self.buf.append((cur, volt, lim, rec["bump"]))

    # ---------------------------------------------------------------- 주기마다
    def _finish_period(self, period):
        """주기가 끝날 때: 평균 전류·평소 대비·전류 상태(2주기 연속 규칙)·배터리 전압·사용량을 기록"""
        cur = np.array([c for c, _, _, _ in self.buf], float)
        volt = np.array([v for _, v, _, _ in self.buf], float)
        cur_ok, volt_ok = cur[~np.isnan(cur)], volt[~np.isnan(volt)]
        out = dict(current=float(np.mean(cur_ok)) if len(cur_ok) else None,
                   voltage=float(np.median(volt_ok)) if len(volt_ok) else None,
                   used_mah=self.used_mah if self.has_current else None, ratio=None, state=None, battery=None)
        # 전류 상태: 처음 5분 기준이 생긴 뒤, 문턱이 아닌 창 중 상한을 넘은 창이 절반을 넘는가
        judged = [(c, lim) for c, _, lim, bump in self.buf if lim is not None and not np.isnan(c) and not bump]
        if judged:
            over = np.mean([c > lim for c, lim in judged]) > 0.5
            lv = int(over)
            if lv != self.state:                           # 진동 판정과 같은 규칙: 2주기 연속이어야 상태를 바꿈
                tgt, cnt = self.pending
                cnt = cnt + 1 if tgt == lv else 1
                self.pending = (lv, cnt)
                if cnt >= 2:
                    self.state, self.pending = lv, (lv, 0)
                    self.events.append((period, f"전류 {STATE_KO[lv]}"))
            else:
                self.pending = (lv, 0)
            out["state"] = STATE_KO[self.state]
            if self.base_mean and self.base_mean > 1e-3 and out["current"] is not None:   # 평소 전류가 거의 0이면 배수를 내지 않음
                out["ratio"] = out["current"] / self.base_mean
        if out["voltage"] is not None:
            low = out["voltage"] < C.BATTERY_LOW_V
            if low and not any(e == "배터리 낮음" for _, e in self.events):
                self.events.append((period, "배터리 낮음"))
            out["battery"] = "낮음" if low else "정상"
        self.periods[period] = out
        self.buf = []

    def finish(self):
        """기록이 끝났을 때 마지막 주기를 마무리"""
        if self.cur_period is not None and self.buf:
            self._finish_period(self.cur_period)

    @property
    def has_data(self):
        return self.has_current or self.has_voltage

    def columns(self, period):
        """workflow.diagnose 표에 붙일 칸 (그 주기 결과가 없으면 빈칸)"""
        p = self.periods.get(period, {})
        r = lambda v, n: round(v, n) if v is not None else None
        cols = {}
        if self.has_current:
            cols.update({"전류(A)": r(p.get("current"), 3), "전류 평소 대비(배)": r(p.get("ratio"), 2),
                         "전류 상태": p.get("state") or "", "사용량(mAh)": r(p.get("used_mah"), 1),
                         "용량 대비 사용(%)": r(p["used_mah"] / C.BATTERY_CAPACITY_MAH * 100, 1) if p.get("used_mah") is not None else None})
        if self.has_voltage:
            cols.update({"배터리 전압(V)": r(p.get("voltage"), 2), "배터리 상태": p.get("battery") or ""})
        return cols
