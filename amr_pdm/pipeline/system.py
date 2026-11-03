"""AMRPipeline: 모든 모델을 하나로 묶은 진단 시스템

    특징 추출 → Ridge(전류 잔차) → 구간별 z-score
                                     ├→ 오토인코더(건강 점수) → 칼만 필터(남은 회차)
                                     └→ 랜덤포레스트 + SHAP(원인·판단 근거)

사용
    pipe = AMRPipeline().fit(data)             # data: {"train_normal": 창 목록, "train_fault": 창 목록, ...}
    pipe.save()
    monitor = AMRPipeline.load().monitor()     # 로봇 1대당 하나
    out = monitor.update(window)               # 1초 창마다 상태·점수·원인·예측 반환
"""
from pathlib import Path

import joblib
import numpy as np

import config as C
from pipeline.features import FEATS
from pipeline.infer import Predictor
from pipeline.models import (AEDetector, CauseClassifier, CurrentModel, IForestDetector, NORMAL_LABEL, PCADetector,
                             SegmentBaseline, ZScoreDetector, calibrate)

BUNDLE_PATH = C.ART_DIR / "model_bundle.joblib"


def per_robot_z(rows):
    """로봇마다 처음 BASELINE_LAPS회로 기준을 만들고, 그 이후 창의 z를 반환"""
    if not rows:
        return [], np.empty((0, len(FEATS)))
    out_rows, Z, robots = [], [], {}
    for r in rows:
        robots.setdefault(r["robot"], []).append(r)
    for rr in robots.values():
        base = SegmentBaseline().fit([r for r in rr if r["lap"] < C.BASELINE_LAPS])
        use = [r for r in rr if r["lap"] >= C.BASELINE_LAPS and not r["bump"]]
        out_rows += use
        Z.append(base.z(use))
    return out_rows, np.vstack(Z)


def split_normal(rows, fit_robots=None, cal_robots=None):
    """정상 창을 모델 학습용과 눈금 보정용으로 나눈다. 지정이 없으면 로봇 1/3을, 로봇이 1대면 뒤쪽 30%를 보정에 쓴다."""
    robots = np.array([r["robot"] for r in rows])
    if fit_robots is not None:
        return np.isin(robots, fit_robots), np.isin(robots, cal_robots)
    names = sorted(set(robots))
    if len(names) >= 2:
        cal = np.isin(robots, names[-max(1, len(names) // 3):])
        return ~cal, cal
    fit = np.arange(len(rows)) < 0.7 * len(rows)
    return fit, ~fit


class AMRPipeline:
    def __init__(self, bundle=None):
        self.bundle = bundle

    def fit(self, data, fit_robots=("N0", "N1", "N2", "N3"), cal_robots=("N4", "N5")):
        """1) 전류 예측 2) 이상 감지 + 건강 점수 눈금 3) 원인 분류 순서로 학습
        고장 데이터(train_fault)가 없으면 이상 감지만 학습하고, 원인 모델은 None이 된다."""
        cur = CurrentModel().fit([r for r in data["train_normal"] if not r["bump"]])
        for rows in data.values():
            if rows:
                cur.add_residuals(rows)

        nrows, Zn = per_robot_z(data["train_normal"])
        fit_mask, cal_mask = split_normal(nrows, fit_robots, cal_robots)
        Zfit, Zcal = Zn[fit_mask], Zn[cal_mask]
        frows, Zf = per_robot_z(data.get("train_fault", []))
        sev = np.array([r["true_sev"] for r in frows], float)
        Zfail, scale = Zf[(sev >= C.FAIL_CAL_BAND[0]) & (sev <= C.FAIL_CAL_BAND[1])], "band"
        if not len(Zfail):                                 # 고장 정도가 없거나 그 구간에 창이 없으면 고장 창 전체로 맞춤
            Zfail, scale = Zf[sev >= C.CAUSE_MIN_SEVERITY], "all_fault" if (sev >= C.CAUSE_MIN_SEVERITY).any() else "normal_only"
        dets = {"zscore": ZScoreDetector(), "iforest": IForestDetector(), "pca": PCADetector(), "ae": AEDetector()}
        for d in dets.values():
            calibrate(d.fit(Zfit), Zcal, Zfail)

        # 원인 분류: 고장 초반(CAUSE_MIN_SEVERITY 미만)은 패턴이 모호하므로 제외, 정상 창은 '정상' 클래스로
        cause = None
        if (sev > 0).any():
            keep = (sev == 0) | (sev >= C.CAUSE_MIN_SEVERITY)
            labels = np.array([NORMAL_LABEL if r["true_sev"] == 0 else f"{r['true_kind']}|{r['true_side']}" for r in frows])
            Zc = np.vstack([Zf[keep], Zcal])
            yc = np.concatenate([labels[keep], [NORMAL_LABEL] * int(cal_mask.sum())])
            cause = CauseClassifier().fit(Zc, yc)

        self.bundle = dict(cur=cur, detectors=dets, cause=cause, feats=FEATS, version="2.0", source="simulation", scale=scale)
        return self

    def save(self, path=BUNDLE_PATH):
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self.bundle, path)
        return path

    @classmethod
    def load(cls, path=BUNDLE_PATH):
        return cls(joblib.load(Path(path)))

    def monitor(self, detector="ae", use_cause=True):
        """로봇 1대를 실시간으로 지켜보는 추론기"""
        return Predictor(self.bundle, detector, use_cause)
