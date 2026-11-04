"""AI 모델 모음 (진동만 사용)

[역할] 파이프라인에 들어가는 모델 부품을 하나씩 정의한다. 학습 순서와 연결은 system.py가, 실시간 사용은 infer.py가 맡는다.
[흐름에서 위치] 진동 특징 36개 → 1(z-score) → 2~6(이상 점수) → health()(건강 점수 0~) → 7(원인) / 8(남은 시간)

[이상 감지 모델의 공통 약속] fit(Z): 정상 z로 학습, raw(Z): 클수록 이상한 점수. 점수 단위가 모델마다 달라서
calibrate()가 모델별로 눈금을 맞추고 health()가 0~ 건강 점수로 바꾼다. 그래서 5개 모델을 같은 기준으로 비교할 수 있다.

1) SegmentBaseline : 구간별 정상 기준(평균·표준편차) → z-score  (복도 지문)
2) ZScoreDetector  : 비교 기준선 — |z| 평균
3) ECODDetector    : ECOD (Li et al., TKDE 2022) — 경험적 분포 꼬리 확률. PyOD가 기본 추천하는 비교 기준
4) IForestDetector : Isolation Forest — PyOD가 기본 추천하는 비교 기준
5) PCADetector     : PCA 복원 오차 — 오토인코더의 선형 비교 대상
6) AEDetector      : 오토인코더(MLP 36-24-6-24-36) — 정상 패턴 복원 오차 = 이상 점수, 건강 점수의 기본 모델
7) CauseClassifier : 랜덤포레스트 + SHAP — 정상/고장 종류(+바퀴) 분류와 판단 근거
8) kalman_forecast : 칼만 필터(수준·기울기) — 정지 위험(9점)까지 남은 시간(주기 수)과 90% 범위
   trend_forecast  : 최근 주기 점수 직선 맞춤 — 칼만 필터의 비교 대상
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import config as C
from pipeline.features import FEATS

Z_CLIP = 20.0                     # z를 ±20에서 자른다 (평소보다 20배 넘게 벗어난 값은 모두 20으로 봄)


# ------------------------------------------------------------ 1. 구간별 정상 기준
class SegmentBaseline:
    """구간(노면·운전 조건)마다 정상 특징의 평균·표준편차를 기억하고, 새 창을 z = (값 - 평균) / 표준편차로 바꾼다.
    엘리베이터 앞처럼 원래 거친 구간에서 진동이 큰 것을 고장으로 오해하지 않게 하는 장치('복도 지문').
    z로 바꾸면 센서 감도·단위가 달라도 '평소보다 몇 배 벗어났나'로 통일되어 뒤의 모델이 다루기 쉽다.
    segment 열이 없으면 전체가 한 구간이다."""

    def fit(self, rows):
        floor = np.full(len(FEATS), 1e-3)                  # 표준편차 하한: 거의 안 변한 특징이 조금만 움직여도 z가 폭발하지 않도록
        use = [r for r in rows if not r["bump"]]           # 문턱을 넘는 순간의 창은 기준에서 뺀다
        if not use:
            raise ValueError("정상 기준을 만들 창이 없습니다.")

        def stat(A):
            """A: 창 × 특징 표 → 특징별 (평균, 표준편차). 표준편차는 '0.001 + 평균의 5%'보다 작게 쓰지 않음"""
            mu, sd = A.mean(0), A.std(0)
            return mu, np.maximum(sd, floor + 0.05 * np.abs(mu))

        # 전체 기준(모든 구간을 합친 것) + 구간별 기준
        self.overall, self.stats = stat(np.array([[r[f] for f in FEATS] for r in use])), {}
        for s in dict.fromkeys(r["segment"] for r in use):  # 나온 순서대로 구간 이름 (중복 없이)
            A = np.array([[r[f] for f in FEATS] for r in use if r["segment"] == s])
            if len(A) >= 5:                                # 창이 너무 적은 구간은 전체 기준을 씀
                self.stats[s] = stat(A)
        return self

    def z(self, rows):
        out = np.empty((len(rows), len(FEATS)))
        for i, r in enumerate(rows):
            mu, sd = self.stats.get(r["segment"], self.overall)   # 이 창의 구간 기준 (학습 때 없던 구간이면 전체 기준)
            out[i] = (np.array([r[f] for f in FEATS]) - mu) / sd
        return np.clip(out, -Z_CLIP, Z_CLIP)                # 극단값 하나가 점수를 독차지하지 않도록 ±20에서 자름


# ------------------------------------------------------------ 2~6. 이상 감지
class ZScoreDetector:
    """가장 단순한 기준: 모든 특징이 평소에서 벗어난 정도의 평균. AI가 이것보다 나아야 의미가 있다."""
    name = "기준선 (z-score)"

    def fit(self, Z):
        return self                                        # 배울 것이 없음 (z 자체가 이미 '평소 대비' 값)

    def raw(self, Z):
        return np.abs(Z).mean(1)                           # 창마다 36개 |z|의 평균


class ECODDetector:
    """특징마다 정상 데이터의 경험적 누적분포로 양쪽 꼬리 확률을 구해 -log 합을 이상 점수로 쓴다 (학습할 파라미터 없음)"""
    name = "ECOD"

    def fit(self, Z):
        self.sorted = np.sort(Z, axis=0)                   # 특징마다 정상 값을 작은 순서로 정렬해 기억 (경험적 분포)
        self.n = len(Z)
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        self.left_skew = np.mean(((Z - mu) / sd) ** 3, axis=0) < 0   # 왜도 < 0: 분포 꼬리가 왼쪽(작은 값)으로 긴 특징
        return self

    def raw(self, Z):
        n = self.n
        # left: 정상 값 중 이 값 이하인 개수, right: 이 값 이상인 개수 (특징마다)
        left = np.column_stack([np.searchsorted(self.sorted[:, j], Z[:, j], side="right") for j in range(Z.shape[1])])
        right = np.column_stack([n - np.searchsorted(self.sorted[:, j], Z[:, j], side="left") for j in range(Z.shape[1])])
        # 꼬리 확률 → -log. 정상에서 거의 안 나오는 값(확률이 작을수록)일수록 큰 수가 된다
        tail_l = -np.log((left + 1) / (n + 1))             # 너무 작은 쪽으로 벗어난 정도
        tail_r = -np.log((right + 1) / (n + 1))            # 너무 큰 쪽으로 벗어난 정도
        auto = np.where(self.left_skew, tail_l, tail_r)    # 특징마다 꼬리가 긴 쪽만 본 값
        # 세 가지(왼쪽 합, 오른쪽 합, 긴 꼬리 합) 중 가장 큰 것이 이 창의 이상 점수
        return np.maximum.reduce([tail_l.sum(1), tail_r.sum(1), auto.sum(1)])


class IForestDetector:
    """무작위로 자르는 나무 100그루에서 적은 횟수로 떨어져 나오는 점일수록 이상. 정상 데이터만으로 학습"""
    name = "Isolation Forest"

    def fit(self, Z):
        self.m = IsolationForest(n_estimators=100, random_state=C.SEED).fit(Z)
        return self

    def raw(self, Z):
        return -self.m.score_samples(Z)                    # scikit-learn은 '정상다울수록 큰 값'을 주므로 부호를 뒤집음


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
        # 주된 방향 6개 (창 수가 적으면 그보다 적게)
        self.m = PCA(n_components=min(6, Z.shape[1], len(Z) - 1), random_state=C.SEED).fit(self.sc.transform(Z))
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        # transform: 6개 방향으로 압축 → inverse_transform: 다시 36개로 펼침 → 원래 값과의 차이(RMSE)
        return np.sqrt(np.mean((self.m.inverse_transform(self.m.transform(X)) - X) ** 2, axis=1))


class AEDetector:
    """주 모델. 신경망이 진동 특징 36개를 6개로 압축했다가 다시 원래 개수로 복원하도록 '정상 데이터로만' 학습한다.
    정상 패턴만 배웠으므로 고장 창은 잘 복원하지 못하고, 복원 오차(RMSE)가 곧 이상 점수가 된다.
    PCA와 달리 특징 사이의 곡선 관계(tanh)도 배울 수 있다. 라벨 없는 정상 데이터만 있어도 학습된다는 것이 장점."""
    name = "오토인코더 (AI)"

    def fit(self, Z):
        self.sc = z_scaler(Z)                              # 신경망은 입력 크기가 고를 때 잘 배우므로 한 번 더 표준화
        X = self.sc.transform(Z)
        # 특징수 → 24 → 6(병목) → 24 → 특징수. 입력과 정답이 같은 X (자기 자신을 복원), 검증 오차가 멈추면 학습 중단
        self.m = MLPRegressor(hidden_layer_sizes=(24, 6, 24), activation="tanh", max_iter=400,
                              early_stopping=True, random_state=C.SEED).fit(X, X)
        return self

    def raw(self, Z):
        X = self.sc.transform(Z)
        return np.sqrt(np.mean((self.m.predict(X) - X) ** 2, axis=1))   # 복원한 값(predict)과 원래 값의 차이(RMSE)


def make_detectors():
    """학습할 이상 감지 모델 5종. 진단은 'ae'(오토인코더)를 쓰고 나머지는 교차검증에서 비교용으로만 쓴다."""
    return {"zscore": ZScoreDetector(), "ecod": ECODDetector(), "iforest": IForestDetector(), "pca": PCADetector(), "ae": AEDetector()}


def calibrate(det, Z_normal, Z_fail):
    """건강 점수 눈금: 정상의 50% 지점 = 0, 99.5% 지점 = 3(주의), 고장 기준 창의 중앙값 = 9(정지 위험)
    고장 데이터가 없으면 rfail = 0이 되어 health()가 정상 눈금을 그대로 늘려 쓴다."""
    r = det.raw(Z_normal)                                  # 학습에 안 쓴 정상 로봇 창들의 이상 점수
    det.r0, det.r995 = float(np.median(r)), float(np.quantile(r, 0.995))   # 정상의 가운데 값, 정상의 상위 0.5% 경계
    det.rfail = float(np.median(det.raw(Z_fail))) if len(Z_fail) else 0.0  # 고장 기준 창들의 가운데 값
    return det


def health(det, Z):
    """정상 99.5% 지점까지는 선형, 그 위는 로그 눈금 (고장 시작 직후 점수가 9를 넘지 않도록)

    고장 기준(rfail)이 정상 99.5% 지점(r995)과 거의 같으면 로그 눈금의 분모가 0에 가까워져 점수가 폭발한다
    (교차검증에서 1e23까지 나옴). 그래서 rfail이 r995의 MIN_FAIL_RATIO배를 넘을 때만 로그 눈금을 쓰고,
    아니면 선형 눈금을 그대로 늘려 쓴다. 점수는 SCORE_MAX에서 자른다."""
    r = det.raw(Z)
    lo, hi = C.THRESH["caution"], C.THRESH["danger"]      # 3점, 9점
    # 직선 눈금: r0 → 0점, r995 → 3점 (분모가 0이 되지 않게 아주 작은 하한을 둠)
    s = lo * (r - det.r0) / max(det.r995 - det.r0, 1e-6 * max(abs(det.r995), 1.0))
    if det.rfail > det.r995 * C.MIN_FAIL_RATIO:
        over = r > det.r995                                # 3점을 넘는 창만
        # 로그 눈금: r995 → 3점, rfail → 9점. 이상 점수가 2배가 될 때마다 건강 점수는 같은 만큼씩 오른다
        s[over] = lo + (hi - lo) * np.log(r[over] / det.r995) / np.log(det.rfail / det.r995)
    return np.clip(s, 0.0, C.SCORE_MAX)                    # 0점 아래, 30점 위는 자름


# ------------------------------------------------------------ 7. 원인 분류
NORMAL_LABEL = "normal|"          # 원인 분류에서 '정상'을 뜻하는 라벨 (종류|바퀴 형식에서 바퀴가 빈 것)


class CauseClassifier:
    """이상이 생겼을 때 '무슨 고장인가'를 답한다. 라벨 예: 'bearing|fr' (종류|바퀴), 'normal|' (정상)
    결정 나무 150그루의 투표 확률을 쓰고, SHAP으로 그 판정에 가장 크게 기여한 특징을 사람이 읽는 이름으로 보여 준다.
    고장 라벨이 있는 데이터가 있어야 학습되며, 없으면 system.py에서 None이 되고 원인 칸이 비게 된다."""

    def fit(self, Z, labels):
        self.m = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                        random_state=C.SEED, n_jobs=-1).fit(Z, labels)
        self.m.n_jobs = 1                                  # 실시간 1건 추론은 단일 스레드가 더 빠름
        self.classes = [str(c) for c in self.m.classes_]   # 배운 라벨 목록 (proba의 열 순서와 같음)
        return self

    def proba(self, Z):
        """창마다 라벨별 확률 (나무 150그루 중 몇 그루가 그 라벨에 투표했나)"""
        return self.m.predict_proba(Z)

    def importance(self):
        """전체적으로 판정에 많이 쓰인 특징 순위 (그래프 fig_importance용)"""
        return sorted(zip(FEATS, self.m.feature_importances_), key=lambda x: -x[1])

    def explain(self, z, label, top=3):
        """SHAP: 이 판정(label)을 가장 크게 밀어 올린 특징 top개 → [(특징, 기여도)]"""
        import shap                                        # 무거운 패키지라 판단 근거가 필요할 때만 불러옴
        if getattr(self, "_explainer", None) is None:
            self._explainer = shap.TreeExplainer(self.m)
        v = self._explainer.shap_values(np.atleast_2d(z), check_additivity=False)   # 특징마다 라벨별 기여도
        i = self.classes.index(label)
        v = v[i][0] if isinstance(v, list) else v[0, :, i]   # shap 버전에 따라 모양이 달라서 둘 다 처리 → 이 라벨의 기여도만
        return [(FEATS[j], float(v[j])) for j in np.argsort(-v)[:top]]   # 기여도가 큰 순서로 top개

    def __getstate__(self):
        """모델 파일로 저장할 때 SHAP 계산기는 빼고 저장 (필요하면 다시 만들 수 있고 파일이 커지기 때문)"""
        return {k: v for k, v in self.__dict__.items() if k != "_explainer"}


def label_ko(lbl):
    """'bearing|fr' → '앞 오른쪽 바퀴 베어링 결함', 'normal|' → '정상' (화면·표에 보이는 이름)"""
    if lbl == NORMAL_LABEL:
        return C.KIND_KO["normal"]
    kind, side = lbl.split("|")
    kind = C.KIND_KO.get(kind, kind)
    return f"{C.WHEEL_KO.get(side, side)} 바퀴 {kind}" if side else kind


# ------------------------------------------------------------ 8. 정지 위험 시점 예측
def kalman_forecast(period_scores, target=C.THRESH["danger"], cfg=None):
    """주기별 점수를 [수준, 기울기] 칼만 필터로 추정해 target까지 남은 시간(주기 수)과 90% 범위를 반환 (상승 추세가 아니면 None)"""
    # 상태 x = [지금 점수 수준, 주기당 기울기]. 매 주기: 예측(수준 += 기울기) → 실제 점수로 보정(칼만 이득 K만큼)
    # q_*: 수준·기울기가 주기마다 얼마나 변할 수 있나, r: 주기 점수 자체의 잡음. 값은 workflow.tune_kalman이 데이터로 맞춤
    cfg = cfg or C.KALMAN
    y = np.asarray(period_scores, float)
    if len(y) < 3:                                         # 점수가 3주기보다 적으면 기울기를 알 수 없음
        return None
    F, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])   # F: 다음 수준 = 수준 + 기울기, H: 눈에 보이는 것은 수준뿐
    Q = np.diag([cfg["q_level"], cfg["q_slope"]])          # 주기마다 수준·기울기가 저절로 변할 수 있는 정도
    x, P = np.array([y[0], 0.0]), np.diag([cfg["r"], 1.0]) # 시작: 수준 = 첫 점수, 기울기 = 0, P = 그 추정이 얼마나 불확실한가
    for v in y[1:]:
        x, P = F @ x, F @ P @ F.T + Q                      # ① 예측: 한 주기 뒤의 수준·기울기 (불확실성은 커짐)
        K = P @ H / (H @ P @ H + cfg["r"])                 # ② 칼만 이득: 실제 점수를 얼마나 믿고 반영할지 (0~1)
        x, P = x + K * (v - H @ x), (np.eye(2) - np.outer(K, H)) @ P   # ③ 보정: 예측과 실제 점수의 차이만큼 고침
    level, slope = float(x[0]), float(x[1])
    if slope <= 0.05:                                      # 주기당 0.05점 이하로 오르면 '나빠지는 중'이 아님
        return None
    out = dict(level=level, slope=slope, n=len(y))
    if level >= target:                                    # 이미 9점 이상
        return dict(out, periods=0.0, periods_lo=0.0, periods_hi=0.0)
    # 남은 주기 수 = (9 - 지금 수준) / 기울기. 기울기의 불확실성(±1.64σ, 90%)으로 빠르면·늦으면 범위를 낸다
    gap, sd = target - level, float(np.sqrt(P[1, 1]))
    return dict(out, periods=gap / slope, periods_lo=gap / (slope + 1.64 * sd),
                periods_hi=gap / (slope - 1.64 * sd) if slope > 1.64 * sd else float("inf"))


def trend_forecast(period_scores, n=C.TREND_PERIODS, target=C.THRESH["danger"]):
    """최근 n주기 건강 점수에 직선을 맞춰 target 도달까지 남은 주기 수를 반환 (상승 추세가 아니면 None)"""
    y = np.asarray(period_scores[-n:], float)              # 최근 n주기 점수만
    if len(y) < 3:
        return None
    x = np.arange(len(y))
    k, b = np.polyfit(x, y, 1)                             # 점수 ≈ k × 주기 + b 인 직선 (k = 주기당 오르는 점수)
    if k <= 0.05:
        return None
    if y[-1] >= target:
        return dict(periods=0.0, slope=float(k), intercept=float(b), n=len(y))
    return dict(periods=float(max(0.0, (target - (k * x[-1] + b)) / k)), slope=float(k), intercept=float(b), n=len(y))
