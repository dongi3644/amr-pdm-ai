"""AI 모델 모음

[역할] 파이프라인에 들어가는 모델 부품을 하나씩 정의한다. 학습 순서와 연결은 system.py가, 실시간 사용은 infer.py가 맡는다.
[흐름에서 위치] 특징(4륜 70개) → 1(전류·배터리 잔차) → 2(z-score) → 3~7(이상 점수) → health()(건강 점수 0~) → 8(원인) / 9(남은 시간)

[이상 감지 모델의 공통 약속] fit(Z): 정상 z로 학습, raw(Z): 클수록 이상한 점수. 점수 단위가 모델마다 달라서
calibrate()가 모델별로 눈금을 맞추고 health()가 0~ 건강 점수로 바꾼다. 그래서 5개 모델을 같은 기준으로 비교할 수 있다.

1) CurrentModel    : Ridge 회귀 — 속도·가속도·구간 → '정상이라면 나와야 할 전류' (센서 불일치 = 실제 - 예측)
   BatteryModel    : Ridge 회귀 — 같은 입력 → '정상이라면 분당 닳아야 할 배터리 %' (초과 소모 = 실제 - 예측)
   BatteryTracker  : 잔량 칸이 떨어진 시각으로 '실제 소모 / 예상 소모' 비율을 재는 장치 (배터리 경고용)
2) SegmentBaseline : 구간별 정상 기준(평균·표준편차) → z-score  (복도 지문)
3) ZScoreDetector  : 비교 기준선 — |z| 평균
4) ECODDetector    : ECOD (Li et al., TKDE 2022) — 경험적 분포 꼬리 확률. PyOD가 기본 추천하는 비교 기준
5) IForestDetector : Isolation Forest — PyOD가 기본 추천하는 비교 기준
6) PCADetector     : PCA 복원 오차 — 오토인코더의 선형 비교 대상
7) AEDetector      : 오토인코더(MLP 특징수-24-6-24-특징수) — 정상 패턴 복원 오차 = 이상 점수, 건강 점수의 기본 모델
8) CauseClassifier : 랜덤포레스트 + SHAP — 정상/고장 종류(+바퀴) 분류와 판단 근거
9) kalman_forecast : 칼만 필터(수준·기울기) — 정지 위험(9점)까지 남은 시간(주기 수)과 90% 범위
   trend_forecast  : 최근 주기 점수 직선 맞춤 — 칼만 필터의 비교 대상
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import config as C
from pipeline.features import FEATS

Z_CLIP = 20.0


# ------------------------------------------------------------ 1. 전류 예측 (센서 불일치)
class CurrentModel:
    """'지금 속도·가속도·구간이면 정상 모터는 전류가 이만큼'을 예측한다.
    실제 전류 - 예측 = 잔차. 잔차가 커지면 같은 주행에 힘이 더 든다는 뜻(마찰·걸림)이라 특징으로 쓴다."""

    def _x(self, rows):                                     # 입력: 구간(원-핫) + 속도 + 가속도 + |가속도| + 속도²
        seg = np.array([[r["segment"] == s for s in self.segments] for r in rows], float)
        sp = np.array([r["speed"] for r in rows])[:, None]
        ac = np.array([r["accel"] for r in rows])[:, None]
        return np.hstack([seg, sp, ac, np.abs(ac), sp ** 2])

    def fit(self, rows):
        self.segments = list(dict.fromkeys(r["segment"] for r in rows))
        X = self._x(rows)
        self.m = {w: Ridge(alpha=1e-3).fit(X, [r[f"{w}_cur"] for r in rows]) for w in C.WHEELS}   # 바퀴마다 모델 하나
        return self

    def add_residuals(self, rows):
        if not rows:
            return rows
        X = self._x(rows)
        pred = {w: self.m[w].predict(X) for w in C.WHEELS}
        for i, r in enumerate(rows):
            for w in C.WHEELS:                              # res_<바퀴> = 실제 전류 - 예상 전류
                r[f"res_{w}"] = r[f"{w}_cur"] - float(pred[w][i])
        return rows


class BatteryModel(CurrentModel):
    """'지금 속도·가속도·구간이면 정상 로봇의 배터리는 분당 몇 % 닳는다'를 예측한다 (입력은 CurrentModel과 같음).
    실제 소모 - 예상 소모 = bat_res (특징으로 쓰임). 주기마다 실제/예상 비율은 infer.py가 따로 보고
    학습한 정상 범위를 넘으면 '배터리 소모 이상'을 낸다. battery 열이 없으면 active=False라 모두 0이 된다."""

    def fit(self, rows):
        self.segments = list(dict.fromkeys(r["segment"] for r in rows))
        y = np.array([r["bat_drain"] for r in rows])
        self.active = bool(np.any(y != 0))
        if self.active:
            self.m = Ridge(alpha=1e-3).fit(self._x(rows), y)
        return self

    def add_residuals(self, rows):
        if not rows:
            return rows
        pred = self.m.predict(self._x(rows)) if self.active else np.zeros(len(rows))
        for r, p in zip(rows, pred):
            r["bat_pred"] = float(max(p, 0.0)) if self.active else 0.0   # 예상 소모는 음수(충전)가 될 수 없음
            r["bat_res"] = r["bat_drain"] - r["bat_pred"] if self.active else 0.0
        return rows


class BatteryTracker:
    """로봇 1대의 배터리 소모 비율을 실시간으로 잰다 (학습 때 경고 기준 정하기와 진단에 같은 코드를 씀).

    비율 = 최근 BATTERY_DROPS칸이 실제로 떨어진 양 / 같은 시간 동안 BatteryModel이 예상한 소모량
        예) 실제 3% 떨어지는 동안 정상이라면 2%만 썼어야 함 → 1.5배 (50% 빨리 닳음)
    측정 구간이 잔량 칸이 바뀐 시각에서 시작하고 끝나므로 1% 계단에 따른 오차가 없다."""

    def __init__(self):
        self.level, self.drops, self.hist, self.cum_exp, self.used = None, [], [], 0.0, 0

    def update(self, row):
        """창 하나: 잔량 칸 변화 기록 + 예상 소모 누적. row에는 battery, t, bat_pred, bump가 있어야 함"""
        if row.get("battery") is None:
            return
        lev = int(np.floor(row["battery"] / C.BATTERY_STEP + 1e-9))
        if self.level is not None and lev > self.level:        # 충전 → 처음부터
            self.drops, self.hist, self.used = [], [], 0
        elif self.level is not None and lev < self.level:
            self.drops += [row["t"]] * (self.level - lev)
        self.level = lev
        if not row["bump"]:
            self.cum_exp += row["bat_pred"] * C.WINDOW_S / 60  # %/분 × 창 길이 → 이 창에서 예상한 소모 %
        self.hist.append((row["t"], self.cum_exp))
        if len(self.drops) > C.BATTERY_DROPS:                   # 오래된 기록은 버림 (측정 구간 시작보다 앞)
            t0 = self.drops[-1 - C.BATTERY_DROPS]
            while len(self.hist) > 1 and self.hist[1][0] <= t0:
                self.hist.pop(0)

    def _exp_at(self, t):
        ts = [h[0] for h in self.hist]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return self.hist[max(i, 0)][1]

    def ratio(self):
        """→ (비율 또는 None, 새로 잰 값인가). 새 칸이 떨어지지 않았으면 직전 값을 다시 주고 fresh=False"""
        if len(self.drops) <= C.BATTERY_DROPS:
            return None, False
        t0, t1 = self.drops[-1 - C.BATTERY_DROPS], self.drops[-1]
        expected = self._exp_at(t1) - self._exp_at(t0)
        if expected < 0.2 * C.BATTERY_DROPS * C.BATTERY_STEP:   # 예상 소모가 거의 없는 구간(거의 정지·충전)은 비율이 무의미
            return None, False
        fresh = len(self.drops) != self.used
        self.used = len(self.drops)
        return C.BATTERY_DROPS * C.BATTERY_STEP / expected, fresh


# ------------------------------------------------------------ 2. 구간별 정상 기준
class SegmentBaseline:
    """구간(노면·운전 조건)마다 정상 특징의 평균·표준편차를 기억하고, 새 창을 z = (값 - 평균) / 표준편차로 바꾼다.
    엘리베이터 앞처럼 원래 거친 구간에서 진동이 큰 것을 고장으로 오해하지 않게 하는 장치('복도 지문').
    z로 바꾸면 센서 종류·단위가 달라도 '평소보다 몇 배 벗어났나'로 통일되어 뒤의 모델이 다루기 쉽다."""

    def fit(self, rows):
        # 표준편차 하한: 정상 기록에서 거의 안 변한 특징(예: 배터리 소모)이 조금만 움직여도 z가 폭발하지 않도록
        floor = np.array([0.01 if f.startswith("bat_") else                # 배터리 소모 [%/분]
                          0.02 if ("cur" in f or f.startswith("res_")) else 1e-3
                          for f in FEATS])
        use = [r for r in rows if not r["bump"]]
        if not use:
            raise ValueError("정상 기준을 만들 창이 없습니다.")

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
        return np.clip(out, -Z_CLIP, Z_CLIP)                # 극단값 하나가 점수를 독차지하지 않도록 ±20에서 자름


# ------------------------------------------------------------ 3~7. 이상 감지
class ZScoreDetector:
    """가장 단순한 기준: 모든 특징이 평소에서 벗어난 정도의 평균. AI가 이것보다 나아야 의미가 있다."""
    name = "기준선 (z-score)"

    def fit(self, Z):
        return self

    def raw(self, Z):
        return np.abs(Z).mean(1)


class ECODDetector:
    """특징마다 정상 데이터의 경험적 누적분포로 양쪽 꼬리 확률을 구해 -log 합을 이상 점수로 쓴다 (학습할 파라미터 없음)"""
    name = "ECOD"

    def fit(self, Z):
        self.sorted = np.sort(Z, axis=0)
        self.n = len(Z)
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        self.left_skew = np.mean(((Z - mu) / sd) ** 3, axis=0) < 0
        return self

    def raw(self, Z):
        n = self.n
        left = np.column_stack([np.searchsorted(self.sorted[:, j], Z[:, j], side="right") for j in range(Z.shape[1])])
        right = np.column_stack([n - np.searchsorted(self.sorted[:, j], Z[:, j], side="left") for j in range(Z.shape[1])])
        tail_l = -np.log((left + 1) / (n + 1))
        tail_r = -np.log((right + 1) / (n + 1))
        auto = np.where(self.left_skew, tail_l, tail_r)
        return np.maximum.reduce([tail_l.sum(1), tail_r.sum(1), auto.sum(1)])


class IForestDetector:
    """무작위로 자르는 나무 100그루에서 적은 횟수로 떨어져 나오는 점일수록 이상. 정상 데이터만으로 학습"""
    name = "Isolation Forest"

    def fit(self, Z):
        self.m = IsolationForest(n_estimators=100, random_state=C.SEED).fit(Z)
        return self

    def raw(self, Z):
        return -self.m.score_samples(Z)


def z_scaler(Z, floor=0.1):
    """z를 다시 표준화하는 StandardScaler. 학습 데이터에서 거의 변하지 않은 특징은 표준편차가 1e-15처럼 잡혀
    새 로봇에서 조금만 달라도 값이 1e14배로 부풀려진다 (교차검증에서 점수 폭발로 확인). z는 이미 '평소 대비 몇 배'
    단위이므로 표준편차를 floor(0.1) 아래로는 쓰지 않는다."""
    sc = StandardScaler().fit(Z)
    sc.scale_ = np.maximum(sc.scale_, floor)
    return sc


class PCADetector:
    """정상 데이터의 주된 방향 6개로만 다시 그려 보고, 다시 그린 것과 원래 값의 차이가 크면 이상 (오토인코더의 직선판)"""
    name = "PCA"

    def fit(self, Z):
        self.sc = z_scaler(Z)
        self.m = PCA(n_components=min(6, Z.shape[1], len(Z) - 1), random_state=C.SEED).fit(self.sc.transform(Z))
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        return np.sqrt(np.mean((self.m.inverse_transform(self.m.transform(X)) - X) ** 2, axis=1))


class AEDetector:
    """주 모델. 신경망이 특징(4륜 70개)을 6개로 압축했다가 다시 원래 개수로 복원하도록 '정상 데이터로만' 학습한다.
    정상 패턴만 배웠으므로 고장 창은 잘 복원하지 못하고, 복원 오차(RMSE)가 곧 이상 점수가 된다.
    PCA와 달리 특징 사이의 곡선 관계(tanh)도 배울 수 있다. 라벨 없는 정상 데이터만 있어도 학습된다는 것이 장점."""
    name = "오토인코더 (AI)"

    def fit(self, Z):
        self.sc = z_scaler(Z)
        X = self.sc.transform(Z)
        # 특징수 → 24 → 6(병목) → 24 → 특징수. 입력과 정답이 같은 X (자기 자신을 복원), 검증 오차가 멈추면 학습 중단
        self.m = MLPRegressor(hidden_layer_sizes=(24, 6, 24), activation="tanh", max_iter=400,
                              early_stopping=True, random_state=C.SEED).fit(X, X)
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        return np.sqrt(np.mean((self.m.predict(X) - X) ** 2, axis=1))


def make_detectors():
    """학습할 이상 감지 모델 5종. 진단은 'ae'(오토인코더)를 쓰고 나머지는 교차검증에서 비교용으로만 쓴다."""
    return {"zscore": ZScoreDetector(), "ecod": ECODDetector(), "iforest": IForestDetector(), "pca": PCADetector(), "ae": AEDetector()}


def calibrate(det, Z_normal, Z_fail):
    """건강 점수 눈금: 정상의 50% 지점 = 0, 99.5% 지점 = 3(주의), 고장 기준 창의 중앙값 = 9(정지 위험)
    고장 데이터가 없으면 rfail = 0이 되어 health()가 정상 눈금을 그대로 늘려 쓴다."""
    r = det.raw(Z_normal)
    det.r0, det.r995 = float(np.median(r)), float(np.quantile(r, 0.995))
    det.rfail = float(np.median(det.raw(Z_fail))) if len(Z_fail) else 0.0
    return det


def health(det, Z):
    """정상 99.5% 지점까지는 선형, 그 위는 로그 눈금 (고장 시작 직후 점수가 9를 넘지 않도록)

    고장 기준(rfail)이 정상 99.5% 지점(r995)과 거의 같으면 로그 눈금의 분모가 0에 가까워져 점수가 폭발한다
    (교차검증에서 1e23까지 나옴). 그래서 rfail이 r995의 MIN_FAIL_RATIO배를 넘을 때만 로그 눈금을 쓰고,
    아니면 선형 눈금을 그대로 늘려 쓴다. 점수는 SCORE_MAX에서 자른다."""
    r = det.raw(Z)
    lo, hi = C.THRESH["caution"], C.THRESH["danger"]
    s = lo * (r - det.r0) / max(det.r995 - det.r0, 1e-6 * max(abs(det.r995), 1.0))
    if det.rfail > det.r995 * C.MIN_FAIL_RATIO:
        over = r > det.r995
        s[over] = lo + (hi - lo) * np.log(r[over] / det.r995) / np.log(det.rfail / det.r995)
    return np.clip(s, 0.0, C.SCORE_MAX)


# ------------------------------------------------------------ 8. 원인 분류
NORMAL_LABEL = "normal|"


class CauseClassifier:
    """이상이 생겼을 때 '무슨 고장인가'를 답한다. 라벨 예: 'bearing|fr' (종류|바퀴), 'normal|' (정상)
    결정 나무 150그루의 투표 확률을 쓰고, SHAP으로 그 판정에 가장 크게 기여한 특징을 사람이 읽는 이름으로 보여 준다.
    고장 라벨이 있는 데이터가 있어야 학습되며, 없으면 system.py에서 None이 되고 원인 칸이 비게 된다."""

    def fit(self, Z, labels):
        self.m = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                        random_state=C.SEED, n_jobs=-1).fit(Z, labels)
        self.m.n_jobs = 1                                  # 실시간 1건 추론은 단일 스레드가 더 빠름
        self.classes = [str(c) for c in self.m.classes_]
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
    kind = C.KIND_KO.get(kind, kind)
    return f"{C.WHEEL_KO.get(side, side)} 바퀴 {kind}" if side else kind


# ------------------------------------------------------------ 9. 정지 위험 시점 예측
def kalman_forecast(period_scores, target=C.THRESH["danger"], cfg=None):
    """주기별 점수를 [수준, 기울기] 칼만 필터로 추정해 target까지 남은 시간(주기 수)과 90% 범위를 반환 (상승 추세가 아니면 None)"""
    # 상태 x = [지금 점수 수준, 주기당 기울기]. 매 주기: 예측(수준 += 기울기) → 실제 점수로 보정(칼만 이득 K만큼)
    # q_*: 수준·기울기가 주기마다 얼마나 변할 수 있나, r: 주기 점수 자체의 잡음. 값은 workflow.tune_kalman이 데이터로 맞춤
    cfg = cfg or C.KALMAN
    y = np.asarray(period_scores, float)
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
        return dict(out, periods=0.0, periods_lo=0.0, periods_hi=0.0)
    # 남은 주기 수 = (9 - 지금 수준) / 기울기. 기울기의 불확실성(±1.64σ, 90%)으로 빠르면·늦으면 범위를 낸다
    gap, sd = target - level, float(np.sqrt(P[1, 1]))
    return dict(out, periods=gap / slope, periods_lo=gap / (slope + 1.64 * sd),
                periods_hi=gap / (slope - 1.64 * sd) if slope > 1.64 * sd else float("inf"))


def trend_forecast(period_scores, n=C.TREND_PERIODS, target=C.THRESH["danger"]):
    """최근 n주기 건강 점수에 직선을 맞춰 target 도달까지 남은 주기 수를 반환 (상승 추세가 아니면 None)"""
    y = np.asarray(period_scores[-n:], float)
    if len(y) < 3:
        return None
    x = np.arange(len(y))
    k, b = np.polyfit(x, y, 1)
    if k <= 0.05:
        return None
    if y[-1] >= target:
        return dict(periods=0.0, slope=float(k), intercept=float(b), n=len(y))
    return dict(periods=float(max(0.0, (target - (k * x[-1] + b)) / k)), slope=float(k), intercept=float(b), n=len(y))
