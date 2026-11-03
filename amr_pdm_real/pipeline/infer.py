"""실시간 추론기: 1초 창이 들어올 때마다 상태·원인·정지 위험 시점을 갱신한다.

[역할] 학습이 끝난 bundle을 받아 로봇 1대를 '지켜보는' 객체(Predictor). 로봇마다 하나씩 만든다.
       창 하나는 흔들리기 쉬우므로, 판정은 판정 주기(기본 1분) 단위로 모아서 하고 상태는 2번 연속일 때만 바꾼다 (오경보 방지).
[흐름에서 위치] bundle(system.py) + 1초 창(dataio) → (여기) update() → 상태·점수·원인·남은 시간
[누가 쓰나] workflow.diagnose(주기별 표), workflow.cross_validate(평가)

흐름
  (per_unit 기준) 처음 BASELINE_PERIODS주기(기본 5분) → 구간별 정상 기준 학습 (이 로봇만의 '복도 지문')
  (reference 기준) 학습 때 만든 공통 기준을 바로 사용
  이후 창마다 → 진동 특징 → z-score → 오토인코더 건강 점수
  판정 주기가 끝날 때마다 → 주기 점수(중앙값) → 상태(정상/주의/경고, 2번 연속일 때만 변경)
                      → 주의 이상이면 칼만 필터로 정지 위험까지 남은 시간과 범위 예측
  주의 이상인 동안 → 랜덤포레스트로 원인 확률 누적, 주기마다 SHAP으로 판단 근거 갱신
"""
from collections import deque

import numpy as np

import config as C
from pipeline.features import FEATS, extract
from pipeline.models import NORMAL_LABEL, SegmentBaseline, health, kalman_forecast, label_ko

FEAT_KO = {"rms": "진동 크기(RMS)", "kurt": "충격성(첨도)", "crest": "크레스트", "low": "저주파 진동",
           "mid": "중주파 진동", "high": "고주파 진동"}


def feat_ko(f):
    """특징 이름 → 화면에 보일 한국어 이름 (SHAP 판단 근거, 그래프 축에 씀)
    fl_rms → '앞 왼쪽 진동 크기(RMS)', dev_fr_kurt → '앞 오른쪽 vs 다른 바퀴: 충격성(첨도)'"""
    parts = f.split("_")
    wheel = lambda w: C.WHEEL_KO.get(w, w)
    if parts[0] == "dev":
        return f"{wheel(parts[1])} vs 다른 바퀴: {FEAT_KO.get(parts[2], parts[2])}"
    return f"{wheel(parts[0])} {FEAT_KO.get(parts[1], parts[1])}"


def level(score):
    return 0 if score < C.THRESH["caution"] else 1 if score < C.THRESH["warning"] else 2


class Predictor:
    def __init__(self, bundle, detector="ae", use_cause=True):
        self.b, self.det = bundle, bundle["detectors"][detector]
        self.use_cause = use_cause and bundle["cause"] is not None   # 고장 라벨 없이 학습하면 원인 모델이 없음
        self.calibrated = bundle["scale"] != "normal_only"           # 고장 데이터가 없으면 남은 시간을 예측하지 않음
        self.reset()

    def reset(self):
        self.base_rows = []
        self.baseline = self.b["reference"] if self.b["baseline"] == "reference" else None
        self.cur_period, self.period_buf, self.period_scores = None, [], []
        self.status, self.pending = 0, (0, 0)            # pending = (목표 상태, 연속 주기 수)
        self.ewma, self.cause_buf, self.forecast = None, deque(maxlen=C.CAUSE_WINDOW), None
        self.period_z, self.reason = [], []
        self.trend_start = 0
        self.events = []                                 # (주기, 상태) 변경 이력
        self.last_cause_label = None

    def _finish_period(self, period):
        """판정 주기가 끝날 때 한 번: 주기 점수 → 상태 변경(2번 연속 규칙) → 남은 시간 예측 → 원인 판단 근거(SHAP) 갱신"""
        if not self.period_buf:
            return
        s = float(np.median(self.period_buf))           # 주기 점수 = 그 주기 창 점수들의 중앙값 (튀는 창 몇 개에 흔들리지 않음)
        self.period_scores.append((period, s))
        self.period_buf = []
        lv = level(s)
        if lv != self.status:
            tgt, cnt = self.pending
            cnt = cnt + 1 if tgt == lv else 1
            self.pending = (lv, cnt)
            if cnt >= 2 or (lv > self.status and s >= C.THRESH["danger"]):
                self.status, self.pending = lv, (lv, 0)
                self.events.append((period, self.status))
                if self.status == 0:                     # 정상 복귀(정비 후): 이전 고장의 원인·추세를 비움
                    self.cause_buf.clear()
                    self.reason, self.trend_start = [], len(self.period_scores)
        else:
            self.pending = (lv, 0)
        scores = [v for _, v in self.period_scores[self.trend_start:]]
        self.forecast = kalman_forecast(scores, cfg=self.b["kalman"]) if self.status >= 1 and self.calibrated else None
        if self.use_cause and self.status >= 1 and self.cause_buf and self.period_z:
            label = self._top_cause()[0]
            if label != NORMAL_LABEL:
                self.reason = [(feat_ko(f), v) for f, v in self.b["cause"].explain(np.mean(self.period_z, axis=0), label)]
        self.period_z = []

    def _top_cause(self):
        p = np.mean(self.cause_buf, axis=0)
        cls = self.b["cause"].classes
        return cls[int(np.argmax(p))], p

    def update(self, rec):
        return self.update_row(extract(rec), rec)

    def update_row(self, row, rec=None):
        """창 하나 처리: (기준 학습 중이면 모으기만) → z → 건강 점수 → 평활 점수 → 원인 확률 누적"""
        if row["period"] != self.cur_period:
            if self.cur_period is not None:
                self._finish_period(self.cur_period)
            self.cur_period = row["period"]
        out = dict(period=row["period"], segment=row["segment"], row=row, rec=rec, phase="monitor",
                   status=self.status, status_ko=C.STATUS_KO[self.status], score=self.ewma,
                   period_scores=list(self.period_scores), forecast=self.forecast, cause=None, cause_probs=None,
                   cause_reason=list(self.reason), top_feats=[], events=list(self.events))

        if self.baseline is None:
            if row["period"] < C.BASELINE_PERIODS:
                if not row["bump"]:
                    self.base_rows.append(row)
                out["phase"] = "learning"
                return out
            self.baseline = SegmentBaseline().fit(self.base_rows)

        if row["bump"]:                                   # 문턱 통과 순간은 판정에서 제외
            out["phase"] = "skip"
            return out

        z = self.baseline.z([row])                          # 특징마다 평소 대비 몇 배 벗어났나
        h = float(health(self.det, z)[0])                   # 오토인코더 복원 오차 → 건강 점수 (3 주의 · 6 경고 · 9 정지 위험)
        a = 2 / (C.EWMA_SPAN + 1)                           # 화면 표시용 지수 이동 평균 (최근 약 20개 창)
        self.ewma = h if self.ewma is None else (1 - a) * self.ewma + a * h
        self.period_buf.append(h)
        self.period_z.append(z[0])
        if self.use_cause and (self.ewma >= 0.8 * C.THRESH["caution"] or self.status >= 1):
            self.cause_buf.append(self.b["cause"].proba(z)[0])
        out.update(score=self.ewma, window_score=h, z=z[0])
        order = np.argsort(-np.abs(z[0]))[:3]
        out["top_feats"] = [(feat_ko(FEATS[i]), float(z[0][i])) for i in order]
        if self.cause_buf and self.status >= 1:
            top, p = self._top_cause()
            cls = self.b["cause"].classes
            out["cause_probs"] = sorted(((label_ko(c), float(v)) for c, v in zip(cls, p)), key=lambda x: -x[1])
            out["cause"] = "원인 미확정" if top == NORMAL_LABEL else label_ko(top)
            out["cause_label"] = self.last_cause_label = top
        return out

    def finish(self):
        if self.cur_period is not None:
            self._finish_period(self.cur_period)
