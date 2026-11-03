"""특징 추출 + 이상 감지 (가장 단순한 기준선 방식)

1) 창마다 진동·전류·온도 특징을 뽑는다
2) 정상 기간 데이터로 '구간별' 정상 기준(평균·표준편차)을 만든다  ← 복도 지문
3) 기준에서 벗어난 정도(z-score)로 건강 점수와 위험도를 계산한다
4) 좌우 바퀴 차이도 함께 본다                                   ← 노면 영향 상쇄

나중에 이 파일만 Isolation Forest, 오토인코더, 1D-CNN 등으로 교체하면 된다.
"""
import numpy as np

import config as C


# ------------------------------------------------------------
# 특징 추출
# ------------------------------------------------------------
def vib_features(x, fs):
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2))
    kurt = np.mean(x ** 4) / (np.mean(x ** 2) ** 2 + 1e-12)
    crest = np.max(np.abs(x)) / (rms + 1e-12)
    spec = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    low = spec[(freqs >= 1) & (freqs < 20)].sum()      # 회전 주파수 근처
    mid = spec[(freqs >= 20) & (freqs < 150)].sum()
    high = spec[freqs >= 150].sum()
    return dict(rms=rms, kurt=kurt, crest=crest, band_low=low, band_mid=mid, band_high=high)


def extract(records):
    """기록 리스트 → 특징 행 리스트"""
    rows = []
    for r in records:
        row = dict(lap=r["lap"], segment=r["segment"], speed=r["speed"], bump=r["bump"])
        for w in ("left", "right"):
            for k, v in vib_features(r[f"{w}_vib"], C.SENSORS["vib_fs"]).items():
                row[f"{w}_{k}"] = v
            row[f"{w}_cur"] = float(np.mean(r[f"{w}_cur"]))
            row[f"{w}_temp"] = r[f"{w}_temp"]
            row[f"{w}_true"] = max(r[f"{w}_true_unbalance"], r[f"{w}_true_looseness"],
                                   r[f"{w}_true_bearing"])
        # 좌우 차이 특징 (노면 진동은 상쇄되고 한쪽 고장만 남음)
        row["diff_band_low"] = row["right_band_low"] - row["left_band_low"]
        row["diff_rms"] = row["right_rms"] - row["left_rms"]
        row["diff_cur"] = row["right_cur"] - row["left_cur"]
        row["diff_temp"] = row["right_temp"] - row["left_temp"]
        rows.append(row)
    return rows


FEATURES = ["left_rms", "right_rms", "left_band_low", "right_band_low", "left_kurt", "right_kurt",
            "left_cur", "right_cur", "diff_band_low", "diff_rms", "diff_cur", "diff_temp"]


# ------------------------------------------------------------
# 이상 감지
# ------------------------------------------------------------
class BaselineDetector:
    """구간별 정상 기준 + z-score 기반 건강 점수"""

    def __init__(self, features=FEATURES):
        self.features = features
        self.stats = {}

    def fit(self, rows):
        """정상 기간(문턱 없는 창)으로 구간별 기준 학습"""
        segs = {r["segment"] for r in rows}
        for s in segs:
            sub = [r for r in rows if r["segment"] == s and not r["bump"]]
            arr = np.array([[r[f] for f in self.features] for r in sub])
            mu, sd = arr.mean(0), arr.std(0)
            # 표준편차가 0에 가까우면 z가 폭주하므로 센서 분해능 수준의 최소값을 둔다
            floor = np.array([C.SENSORS["temp_res"] if "temp" in f else
                              0.02 if "cur" in f else 1e-3 for f in self.features])
            self.stats[s] = (mu, np.maximum(sd, floor + 0.05 * np.abs(mu)))

    def score(self, row):
        mu, sd = self.stats[row["segment"]]
        z = (np.array([row[f] for f in self.features]) - mu) / sd
        z = np.clip(np.abs(z), 0, 20)           # 한 특징이 점수를 독차지하지 않도록 상한
        return float(np.mean(z))                # 클수록 정상에서 멂

    @staticmethod
    def risk(score):
        if score < 3:
            return "정상"
        if score < 6:
            return "주의"
        return "경고"
