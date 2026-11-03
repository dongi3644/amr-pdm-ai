"""특징 추출: 1초 창의 원시 신호 → 숫자 특징

[역할] 수천 개 숫자로 된 진동 파형을, 고장과 관련 있다고 알려진 요약 숫자로 바꾼다. 모든 AI 모델은 이 숫자만 본다.
[흐름에서 위치] dataio의 1초 창 → (여기) → models.CurrentModel·BatteryModel(잔차 추가) → SegmentBaseline(z-score) → 이상 감지·원인 분류

[특징이 잡아내는 고장 신호]
    RMS(진동 크기)    전반적으로 흔들림이 커짐 — 불평형, 풀림
    첨도·크레스트      뾰족한 충격이 섞임 — 베어링 표면 손상의 초기 신호
    저·중·고주파 에너지 어느 주파수 대역이 커졌는가 — 고장 종류마다 다름
    1X 진폭           바퀴가 한 바퀴 돌 때마다 반복되는 진동 — 불평형의 대표 신호
    전류              마찰·부하가 늘면 모터 전류가 오름
    배터리 소모 속도    같은 주행인데 평소보다 빨리 닳음 — 구동부 저항 증가, 배터리 열화
    (온도는 쓰지 않는다)

[특징 목록] 바퀴 4개 (앞 왼쪽 fl, 앞 오른쪽 fr, 뒤 왼쪽 rl, 뒤 오른쪽 rr)
    바퀴마다 8개 (32개) 진동 7개(RMS, 첨도, 크레스트, 저 1~20 Hz·중 20~150 Hz·고 150 Hz 이상 에너지, 1X 진폭) + 전류
    바퀴마다 8개 (32개) '이 바퀴 - 나머지 세 바퀴 평균' (dev_). 노면처럼 모든 바퀴에 같이 오는 영향은 빠지고
                        한 바퀴에만 생긴 고장이 남아, 고장 난 바퀴의 위치를 짚을 수 있다
    배터리 1개          소모 속도 [%/분] (로봇 전체에 하나)
    잔차 5개            바퀴별 전류 불일치 4개(models.CurrentModel) + 배터리 초과 소모 1개(models.BatteryModel)
    합계 = 32 + 32 + 1 + 5 = 70개
"""
import numpy as np

import config as C

WHEELS = tuple(C.WHEELS)
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high", "x1"]
WHEEL_KEYS = VIB_KEYS + ["cur"]                        # 바퀴마다 계산하는 특징 (바퀴 사이 비교도 이 목록으로)


def vib_features(x, fs, rot_hz):
    """진동 파형 1초 → 7개. x: 파형, fs: 샘플링 [Hz], rot_hz: 바퀴 회전 주파수 [Hz]"""
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
        x1=float(spec[(f >= rot_hz - 1.01) & (f <= rot_hz + 1.01)].max(initial=0.0)),   # 차수 추적(1X)
    )


def extract(rec):
    """1초 창(dataio.load가 만든 dict) → 특징 dict. 라벨은 평가용으로만 옮겨 담고 모델 입력에는 쓰지 않는다."""
    rot_hz = rec["speed"] / C.WHEEL_RADIUS / (2 * np.pi)   # 속도 [m/s] → 바퀴 회전 주파수 [Hz]
    row = dict(period=rec["period"], segment=rec["segment"], speed=rec["speed"], accel=rec["accel"], bump=rec["bump"])
    for w in WHEELS:
        for k, v in vib_features(rec[f"{w}_vib"], rec["vib_fs"], rot_hz).items():
            row[f"{w}_{k}"] = v
        row[f"{w}_cur"] = float(rec[f"{w}_cur"])
    if len(WHEELS) > 1:                                    # 바퀴 사이 비교: 이 바퀴 - 나머지 바퀴들 평균
        for k in WHEEL_KEYS:
            vals = np.array([row[f"{w}_{k}"] for w in WHEELS])
            others = (vals.sum() - vals) / (len(WHEELS) - 1)
            for w, v, o in zip(WHEELS, vals, others):
                row[f"dev_{w}_{k}"] = float(v - o)
    row["bat_drain"] = float(rec["bat_drain"])             # 배터리 소모 속도 [%/분] (battery 열이 없으면 0)
    row["battery"], row["t"] = rec["battery"], rec["t"]    # 잔량·시각: 특징은 아니고 배터리 소모 비율(BatteryTracker)에 씀
    row.update(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"], labeled=rec["labeled"])
    return row


BASE_FEATS = ([f"{w}_{k}" for w in WHEELS for k in WHEEL_KEYS]
              + ([f"dev_{w}_{k}" for w in WHEELS for k in WHEEL_KEYS] if len(WHEELS) > 1 else [])
              + ["bat_drain"])
FEATS = BASE_FEATS + [f"res_{w}" for w in WHEELS] + ["bat_res"]
