"""특징 추출: 1초 창의 진동 파형 → 숫자 특징 36개

[역할] 수백~수천 개 숫자로 된 진동 파형을, 고장과 관련 있다고 알려진 요약 숫자로 바꾼다. 모든 진동 AI 모델은 이 숫자만 본다.
[흐름에서 위치] dataio의 1초 창 → (여기) → SegmentBaseline(z-score) → 이상 감지·원인 분류
[쓰는 데이터] 차체에 단 MPU6050 하나의 6축 (가속도 ax·ay·az, 각속도 gx·gy·gz).
              속도가 없으므로 회전 주파수(1X) 특징은 계산하지 않는다 (속도를 가정하면 임의 값이 되므로).
              전류·전압은 학습 데이터에 없으므로 여기(진동 AI)에 넣지 않고 power.py가 따로 감시한다.

[특징이 잡아내는 고장 신호]
    RMS(진동 크기)    전반적으로 흔들림이 커짐 — 불평형, 풀림
    첨도·크레스트      뾰족한 충격이 섞임 — 베어링·기어 표면 손상의 초기 신호
    저·중·고주파 에너지 어느 주파수 대역이 커졌는가 — 고장 종류마다 다름
    가속도 vs 각속도   가속도는 '떨림', 각속도는 '차체가 흔들리며 기우는 것' — 같은 이상도 두 센서에 다르게 드러나 서로 보완함

[특징 목록] 축 6개 × 6개 = 36개
    ax_rms, ax_kurt, ax_crest, ax_low, ax_mid, ax_high, ay_rms, ..., gz_high
    저(1~20 Hz)·중(20~150 Hz)·고(150 Hz 이상) 주파수 에너지. 샘플링이 300 Hz보다 낮으면 고주파 특징은 거의 비게 된다
"""
import numpy as np

import config as C

CHANNELS = tuple(C.VIB_CHANNELS)
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high"]


def vib_features(x, fs):
    """진동 파형 1초 → 6개. x: 파형, fs: 샘플링 [Hz]"""
    x = np.asarray(x, float) - np.mean(x)              # 중력·센서 오프셋·일정한 회전(평균)을 빼고 흔들림만 남김
    ms = np.mean(x ** 2) + 1e-12                       # 평균 제곱 (진동 에너지). 아주 작은 수를 더해 0으로 나누는 것을 막음
    spec = np.abs(np.fft.rfft(x)) / len(x) * 2         # FFT 진폭 스펙트럼 (1초 창이라 주파수 간격 1 Hz)
    f = np.fft.rfftfreq(len(x), 1 / fs)                # spec의 칸마다 몇 Hz인가 (0, 1, 2, ... Hz)
    return dict(
        rms=float(np.sqrt(ms)),                        # 진동 크기: 평균 제곱의 제곱근
        kurt=float(np.mean(x ** 4) / ms ** 2),         # 첨도: 고르게 흔들리면 약 3, 뾰족한 충격이 섞이면 커짐
        crest=float(np.max(np.abs(x)) / np.sqrt(ms)),  # 크레스트: 가장 큰 순간값이 평균 크기의 몇 배인가
        low=float(spec[(f >= 1) & (f < 20)].sum()),    # 1~20 Hz 진폭 합 (느린 흔들림. 주로 불평형·풀림)
        mid=float(spec[(f >= 20) & (f < 150)].sum()),  # 20~150 Hz 진폭 합
        high=float(spec[f >= 150].sum()),              # 150 Hz 이상 진폭 합 (빠른 떨림. 주로 베어링 손상)
    )


def extract(rec):
    """1초 창(dataio.load가 만든 dict) → 특징 dict. 라벨은 평가용으로만 옮겨 담고 모델 입력에는 쓰지 않는다."""
    row = dict(period=rec["period"], segment=rec["segment"], bump=rec["bump"])
    for ch in CHANNELS:                                    # 축마다 6개 → ax_rms, ax_kurt, ..., gz_high (36개)
        for k, v in vib_features(rec[ch], rec["vib_fs"]).items():
            row[f"{ch}_{k}"] = v
    row.update(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"], labeled=rec["labeled"])
    return row


# 모델에 들어가는 특징 이름 36개와 그 순서. 모든 모델이 이 순서대로 숫자를 받으므로 순서를 바꾸면 저장된 모델과 맞지 않게 된다
FEATS = [f"{ch}_{k}" for ch in CHANNELS for k in VIB_KEYS]
