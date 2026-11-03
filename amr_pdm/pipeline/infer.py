"""실시간 추론기: 1초 창이 들어올 때마다 상태·원인·정지 위험 시점을 갱신한다.

흐름
  처음 BASELINE_LAPS회 주행 → 구간별 정상 기준 학습 (이 로봇만의 '복도 지문')
  이후 창마다 → 특징 → 전류 잔차 → z-score → 오토인코더 건강 점수
  회차가 끝날 때마다 → 회차 점수(중앙값) → 상태(정상/주의/경고, 2회 연속일 때만 변경)
                      → 주의 이상이면 칼만 필터로 정지 위험까지 남은 회차와 범위 예측
  주의 이상인 동안 → 랜덤포레스트로 원인(종류·바퀴) 확률 누적, 회차마다 SHAP으로 판단 근거 갱신
"""
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import config as C
from pipeline.features import FEATS, extract
from pipeline.models import NORMAL_LABEL, SegmentBaseline, health, kalman_forecast, label_ko

FEAT_KO = {
    "rms": "진동 크기(RMS)", "kurt": "충격성(첨도)", "crest": "크레스트", "low": "저주파 진동",
    "mid": "중주파 진동", "high": "고주파 진동", "x1": "회전주파수(1X) 진동", "cur": "모터 전류",
    "temp": "모터 온도", "res": "전류 불일치(예측 대비)",
}


def feat_ko(f):
    side, key = f.split("_", 1)
    pre = {"left": "좌측 ", "right": "우측 ", "diff": "좌우 차이: "}.get(side, "")
    return pre + FEAT_KO.get(key, key)


def level(score):
    return 0 if score < C.THRESH["caution"] else 1 if score < C.THRESH["warning"] else 2


class Predictor:
    def __init__(self, bundle, detector="ae", use_cause=True):
        self.b, self.det = bundle, bundle["detectors"][detector]
        self.use_cause = use_cause and bundle["cause"] is not None   # 고장 라벨 없이 학습하면 원인 모델이 없음
        self.calibrated = bundle.get("scale") != "normal_only"       # 고장 데이터가 없으면 9점이 실제 고장과 맞지 않아 남은 회차를 예측하지 않음
        self.reset()

    @classmethod
    def load(cls, path=C.ART_DIR / "model_bundle.joblib", detector="ae"):
        return cls(joblib.load(Path(path)), detector)

    def reset(self):
        self.base_rows, self.baseline = [], None
        self.cur_lap, self.lap_buf, self.lap_scores = 0, [], []
        self.status, self.pending = 0, (0, 0)            # pending = (목표 상태, 연속 회차 수)
        self.ewma, self.cause_buf, self.forecast = None, deque(maxlen=C.CAUSE_WINDOW), None
        self.lap_z, self.reason = [], []                 # 이번 회차 z, 원인 판단 근거(SHAP)
        self.trend_start = 0                             # 추세 추정에 쓰는 첫 회차 위치 (정상 복귀 시 갱신)
        self.events = []                                 # (회차, 상태) 변경 이력

    # ---------------------------------------------------------------- 회차 마감
    def _finish_lap(self, lap):
        if not self.lap_buf:
            return
        s = float(np.median(self.lap_buf))
        self.lap_scores.append((lap, s))
        self.lap_buf = []
        lv = level(s)
        if lv != self.status:
            tgt, cnt = self.pending
            cnt = cnt + 1 if tgt == lv else 1
            self.pending = (lv, cnt)
            if cnt >= 2 or (lv > self.status and s >= C.THRESH["danger"]):
                self.status, self.pending = lv, (lv, 0)
                self.events.append((lap, self.status))
                if self.status == 0:                     # 정상 복귀(정비 후): 이전 고장의 원인·추세를 비움
                    self.cause_buf.clear()
                    self.reason, self.trend_start = [], len(self.lap_scores)
        else:
            self.pending = (lv, 0)
        scores = [v for _, v in self.lap_scores[self.trend_start:]]
        self.forecast = kalman_forecast(scores) if self.status >= 1 and self.calibrated else None
        if self.use_cause and self.status >= 1 and self.cause_buf and self.lap_z:
            label = self._top_cause()[0]
            if label != NORMAL_LABEL:
                self.reason = [(feat_ko(f), v) for f, v in self.b["cause"].explain(np.mean(self.lap_z, axis=0), label)]
        self.lap_z = []

    def _top_cause(self):
        p = np.mean(self.cause_buf, axis=0)
        cls = self.b["cause"].classes
        return cls[int(np.argmax(p))], p

    # ---------------------------------------------------------------- 창 하나 처리
    def update(self, rec):
        return self.update_row(extract(rec), rec)

    def update_row(self, row, rec=None):
        self.b["cur"].add_residuals([row])
        if row["lap"] != self.cur_lap:
            self._finish_lap(self.cur_lap)
            self.cur_lap = row["lap"]
        out = dict(lap=row["lap"], segment=row["segment"], row=row, rec=rec, phase="monitor",
                   status=self.status, status_ko=C.STATUS_KO[self.status], score=self.ewma,
                   lap_scores=list(self.lap_scores), forecast=self.forecast, cause=None, cause_probs=None,
                   cause_reason=list(self.reason), top_feats=[], events=list(self.events))

        if self.baseline is None:
            if row["lap"] < C.BASELINE_LAPS:
                if not row["bump"]:
                    self.base_rows.append(row)
                out["phase"] = "learning"
                return out
            self.baseline = SegmentBaseline().fit(self.base_rows)

        if row["bump"]:                                   # 문턱 통과 순간은 판정에서 제외
            out["phase"] = "skip"
            return out

        z = self.baseline.z([row])
        h = float(health(self.det, z)[0])
        a = 2 / (C.EWMA_SPAN + 1)
        self.ewma = h if self.ewma is None else (1 - a) * self.ewma + a * h
        self.lap_buf.append(h)
        self.lap_z.append(z[0])
        if self.use_cause and (self.ewma >= 0.8 * C.THRESH["caution"] or self.status >= 1):
            self.cause_buf.append(self.b["cause"].proba(z)[0])
        out.update(score=self.ewma, window_score=h, z=z[0])
        order = np.argsort(-np.abs(z[0]))[:3]
        out["top_feats"] = [(feat_ko(FEATS[i]), float(z[0][i])) for i in order]
        if self.cause_buf and self.status >= 1:
            top, p = self._top_cause()
            cls = self.b["cause"].classes
            out["cause_probs"] = sorted(((label_ko(c), float(v)) for c, v in zip(cls, p)), key=lambda x: -x[1])
            out["cause"] = "원인 미확정" if top == NORMAL_LABEL else label_ko(top)   # 이상은 떴지만 아는 고장 패턴이 아닌 경우
            out["cause_label"] = top
        return out

    def finish(self):
        self._finish_lap(self.cur_lap)
