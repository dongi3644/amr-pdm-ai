"""AI 모델 모음

1) CurrentModel    : 속도·가속도·구간 → '정상이라면 나와야 할 전류' 예측 (센서 불일치 = 실제 - 예측)
2) SegmentBaseline : 로봇마다 구간별 정상 기준(평균·표준편차) → z-score  (복도 지문)
3) ZScoreDetector  : 비교 기준선 — |z| 평균
4) IForestDetector : Isolation Forest — 정상 데이터만으로 학습하는 이상 감지
5) PCADetector     : PCA 복원 오차 — 오토인코더의 선형 비교 대상
6) AEDetector      : 오토인코더(MLP) — 정상 패턴 복원 오차 = 이상 점수, 건강 점수의 기본 모델
7) CauseClassifier : 랜덤포레스트 + SHAP — 정상/고장 종류 + 바퀴(좌/우) 분류와 판단 근거
8) kalman_forecast : 칼만 필터(선형 추세) — 건강 점수 평활, 정지 위험(9점)까지 남은 회차와 예측 범위
   trend_forecast  : 최근 회차 직선 맞춤 — 칼만 필터의 비교 대상
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import config as C
from pipeline.features import BASE_FEATS, FEATS

Z_CLIP = 20.0


# ------------------------------------------------------------ 1. 전류 예측 (센서 불일치)
class CurrentModel:
    def _x(self, rows):
        seg = np.array([[r["segment"] == s for s in self.segments] for r in rows], float)
        sp = np.array([r["speed"] for r in rows])[:, None]
        ac = np.array([r["accel"] for r in rows])[:, None]
        return np.hstack([seg, sp, ac, np.abs(ac), sp ** 2])

    def fit(self, rows):
        self.segments = list(dict.fromkeys(r["segment"] for r in rows))
        X = self._x(rows)
        self.m = {w: Ridge(alpha=1e-3).fit(X, [r[f"{w}_cur"] for r in rows]) for w in ("left", "right")}
        return self

    def add_residuals(self, rows):
        X = self._x(rows)
        pred = {w: self.m[w].predict(X) for w in ("left", "right")}
        for i, r in enumerate(rows):
            r["pred_left"], r["pred_right"] = float(pred["left"][i]), float(pred["right"][i])
            r["res_left"] = r["left_cur"] - r["pred_left"]
            r["res_right"] = r["right_cur"] - r["pred_right"]
            r["diff_res"] = r["res_right"] - r["res_left"]
        return rows


# ------------------------------------------------------------ 2. 구간별 정상 기준
class SegmentBaseline:
    def fit(self, rows):
        floor = np.array([C.SENSORS["temp_res"] if "temp" in f else
                          0.02 if ("cur" in f or f.startswith("res_") or f == "diff_res") else 1e-3
                          for f in FEATS])
        use = [r for r in rows if not r["bump"]]

        def stat(A):
            mu, sd = A.mean(0), A.std(0)
            return mu, np.maximum(sd, floor + 0.05 * np.abs(mu))

        self.overall, self.stats = stat(np.array([[r[f] for f in FEATS] for r in use])), {}
        for s in dict.fromkeys(r["segment"] for r in use):
            A = np.array([[r[f] for f in FEATS] for r in use if r["segment"] == s])
            if len(A) >= 5:                                # 창이 너무 적은 구간은 전체 기준을 씀
                self.stats[s] = stat(A)
        return self

    def z(self, rows):
        out = np.empty((len(rows), len(FEATS)))
        for i, r in enumerate(rows):
            mu, sd = self.stats.get(r["segment"], self.overall)
            out[i] = (np.array([r[f] for f in FEATS]) - mu) / sd
        return np.clip(out, -Z_CLIP, Z_CLIP)


# ------------------------------------------------------------ 3~6. 이상 감지
class ZScoreDetector:
    name = "기준선 (z-score)"

    def fit(self, Z):
        return self

    def raw(self, Z):
        return np.abs(Z).mean(1)


class IForestDetector:
    name = "Isolation Forest"

    def fit(self, Z):
        self.m = IsolationForest(n_estimators=100, random_state=C.SEED).fit(Z)
        return self

    def raw(self, Z):
        return -self.m.score_samples(Z)


class PCADetector:
    name = "PCA"

    def fit(self, Z):
        self.sc = StandardScaler().fit(Z)
        self.m = PCA(n_components=6, random_state=C.SEED).fit(self.sc.transform(Z))
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        return np.sqrt(np.mean((self.m.inverse_transform(self.m.transform(X)) - X) ** 2, axis=1))


class AEDetector:
    name = "오토인코더 (AI)"

    def fit(self, Z):
        self.sc = StandardScaler().fit(Z)
        X = self.sc.transform(Z)
        self.m = MLPRegressor(hidden_layer_sizes=(24, 6, 24), activation="tanh", max_iter=400,
                              early_stopping=True, random_state=C.SEED).fit(X, X)
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        return np.sqrt(np.mean((self.m.predict(X) - X) ** 2, axis=1))


def calibrate(det, Z_normal, Z_fail):
    """건강 점수 눈금 맞춤: 정상의 50% 지점 = 0, 99.5% 지점 = 3(주의), 고장 정도 0.8 창의 중앙값 = 9(정지 위험)
    고장 데이터가 없으면 rfail = 0이 되어 health()가 정상 눈금을 그대로 늘려 쓴다."""
    r = det.raw(Z_normal)
    det.r0, det.r995 = float(np.median(r)), float(np.quantile(r, 0.995))
    det.rfail = float(np.median(det.raw(Z_fail))) if len(Z_fail) else 0.0
    return det


def health(det, Z):
    """정상 99.5% 지점까지는 선형, 그 위는 로그 눈금 (고장 시작 직후 점수가 9를 넘지 않도록)"""
    r = det.raw(Z)
    lo, hi = C.THRESH["caution"], C.THRESH["danger"]
    s = lo * (r - det.r0) / (det.r995 - det.r0 + 1e-12)
    if det.rfail > det.r995:
        over = r > det.r995
        s[over] = lo + (hi - lo) * np.log(r[over] / det.r995) / np.log(det.rfail / det.r995)
    return np.maximum(0.0, s)


# ------------------------------------------------------------ 7. 원인 분류
NORMAL_LABEL = "normal|"


class CauseClassifier:
    def fit(self, Z, labels):
        self.m = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                        random_state=C.SEED, n_jobs=-1).fit(Z, labels)
        self.m.n_jobs = 1                                  # 실시간 1건 추론은 단일 스레드가 더 빠름
        self.classes = list(self.m.classes_)
        return self

    def proba(self, Z):
        return self.m.predict_proba(Z)

    def importance(self):
        return sorted(zip(FEATS, self.m.feature_importances_), key=lambda x: -x[1])

    def explain(self, z, label, top=3):
        """SHAP: 이 판정(label)을 가장 크게 밀어 올린 특징 top개 → [(특징, 기여도)]"""
        import shap                                        # 무거운 패키지라 판단 근거가 필요할 때만 불러옴
        if getattr(self, "_explainer", None) is None:
            self._explainer = shap.TreeExplainer(self.m)
        v = self._explainer.shap_values(np.atleast_2d(z), check_additivity=False)
        i = self.classes.index(label)
        v = v[i][0] if isinstance(v, list) else v[0, :, i]
        return [(FEATS[j], float(v[j])) for j in np.argsort(-v)[:top]]

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_explainer"}


def label_ko(lbl):
    if lbl == NORMAL_LABEL:
        return C.KIND_KO["normal"]
    kind, side = lbl.split("|")
    kind = C.KIND_KO.get(kind, kind)                       # 사용자 데이터는 고장 이름이 자유롭고, 센서가 한쪽뿐이면 바퀴가 없음
    return f"{C.SIDE_KO[side]} 바퀴 {kind}" if side in C.SIDE_KO else kind


# ------------------------------------------------------------ 8. 정지 위험 시점 예측
def kalman_forecast(lap_scores, target=C.THRESH["danger"], cfg=C.KALMAN):
    """회차 점수를 [수준, 기울기] 칼만 필터로 추정해 target까지 남은 회차와 90% 범위를 반환 (상승 추세가 아니면 None)"""
    y = np.asarray(lap_scores, float)
    if len(y) < 3:
        return None
    F, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])
    Q = np.diag([cfg["q_level"], cfg["q_slope"]])
    x, P = np.array([y[0], 0.0]), np.diag([cfg["r"], 1.0])
    for v in y[1:]:
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H / (H @ P @ H + cfg["r"])
        x, P = x + K * (v - H @ x), (np.eye(2) - np.outer(K, H)) @ P
    level, slope = float(x[0]), float(x[1])
    if slope <= 0.05:
        return None
    out = dict(level=level, slope=slope, n=len(y))
    if level >= target:
        return dict(out, laps=0.0, laps_lo=0.0, laps_hi=0.0)
    gap, sd = target - level, float(np.sqrt(P[1, 1]))
    return dict(out, laps=gap / slope, laps_lo=gap / (slope + 1.64 * sd),
                laps_hi=gap / (slope - 1.64 * sd) if slope > 1.64 * sd else float("inf"))


def trend_forecast(lap_scores, n=C.TREND_LAPS, target=C.THRESH["danger"]):
    """최근 n회차 건강 점수에 직선을 맞춰 target 도달까지 남은 회차 수를 반환 (상승 추세가 아니면 None)"""
    y = np.asarray(lap_scores[-n:], float)
    if len(y) < 3:
        return None
    x = np.arange(len(y))
    k, b = np.polyfit(x, y, 1)
    if k <= 0.05:
        return None
    if y[-1] >= target:
        return dict(laps=0.0, slope=float(k), intercept=float(b), n=len(y))
    return dict(laps=float(max(0.0, (target - (k * x[-1] + b)) / k)), slope=float(k), intercept=float(b), n=len(y))
