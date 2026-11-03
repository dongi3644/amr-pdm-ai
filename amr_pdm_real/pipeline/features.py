"""특징 추출: 1초 창의 진동 파형 → 숫자 특징 48개

[역할] 수천 개 숫자로 된 진동 파형을, 고장과 관련 있다고 알려진 요약 숫자로 바꾼다. 모든 AI 모델은 이 숫자만 본다.
[흐름에서 위치] dataio의 1초 창 → (여기) → SegmentBaseline(z-score) → 이상 감지·원인 분류
[쓰는 데이터] 바퀴 4개의 진동만. 속도가 없으므로 회전 주파수(1X) 특징은 계산하지 않는다 (속도를 가정하면 임의 값이 되므로).

[특징이 잡아내는 고장 신호]
    RMS(진동 크기)    전반적으로 흔들림이 커짐 — 불평형, 풀림
    첨도·크레스트      뾰족한 충격이 섞임 — 베어링 표면 손상의 초기 신호
    저·중·고주파 에너지 어느 주파수 대역이 커졌는가 — 고장 종류마다 다름

[특징 목록] 바퀴 4개 (앞 왼쪽 fl, 앞 오른쪽 fr, 뒤 왼쪽 rl, 뒤 오른쪽 rr)
    바퀴마다 6개 (24개) RMS, 첨도, 크레스트, 저(1~20 Hz)·중(20~150 Hz)·고(150 Hz 이상) 주파수 에너지
    바퀴마다 6개 (24개) '이 바퀴 - 나머지 세 바퀴 평균' (dev_). 노면처럼 모든 바퀴에 같이 오는 영향은 빠지고
                        한 바퀴에만 생긴 고장이 남아, 고장 난 바퀴의 위치를 짚을 수 있다
    합계 48개
"""
import numpy as np

import config as C

WHEELS = tuple(C.WHEELS)
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high"]


def vib_features(x, fs):
    """진동 파형 1초 → 6개. x: 파형, fs: 샘플링 [Hz]"""
    x = np.asarray(x, float) - np.mean(x)              # 중력·센서 오프셋(평균)을 빼고 흔들림만 남김
    ms = np.mean(x ** 2) + 1e-12
    spec = np.abs(np.fft.rfft(x)) / len(x) * 2         # FFT 진폭 스펙트럼 (1초 창이라 주파수 간격 1 Hz)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return dict(
        rms=float(np.sqrt(ms)),
        kurt=float(np.mean(x ** 4) / ms ** 2),
        crest=float(np.max(np.abs(x)) / np.sqrt(ms)),
        low=float(spec[(f >= 1) & (f < 20)].sum()),
        mid=float(spec[(f >= 20) & (f < 150)].sum()),
        high=float(spec[f >= 150].sum()),
    )


def extract(rec):
    """1초 창(dataio.load가 만든 dict) → 특징 dict. 라벨은 평가용으로만 옮겨 담고 모델 입력에는 쓰지 않는다."""
    row = dict(period=rec["period"], segment=rec["segment"], bump=rec["bump"])
    for w in WHEELS:
        for k, v in vib_features(rec[f"{w}_vib"], rec["vib_fs"]).items():
            row[f"{w}_{k}"] = v
    for k in VIB_KEYS:                                     # 바퀴 사이 비교: 이 바퀴 - 나머지 세 바퀴 평균
        vals = np.array([row[f"{w}_{k}"] for w in WHEELS])
        others = (vals.sum() - vals) / (len(WHEELS) - 1)
        for w, v, o in zip(WHEELS, vals, others):
            row[f"dev_{w}_{k}"] = float(v - o)
    row.update(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"], labeled=rec["labeled"])
    return row


FEATS = [f"{w}_{k}" for w in WHEELS for k in VIB_KEYS] + [f"dev_{w}_{k}" for w in WHEELS for k in VIB_KEYS]
