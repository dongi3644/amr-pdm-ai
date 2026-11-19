"""가상 이상 넣기: 처음 보는 고장에 대비한 학습 보조 (config.SYNTH_ANOMALY, 11/19)

[역할] 랜덤포레스트는 '배운 고장'만 잘 알아본다. 그래서 학습용 정상 로봇의 진동 파형에 '어떤 고장이든 생길 수 있는 모양'을
       무작위로 넣은 복사본을 만들어 '처음 보는 이상(unknown)' 라벨로 함께 가르친다 (outlier exposure 방식).
       바닥이 거칠어지면 진동이 넓은 주파수에서 함께 커지지만(정상), 고장은 특정 주파수·특정 박자로 커진다는 차이를 배우게 한다.
[흐름에서 위치] workflow.train: 정상 로봇 창 → (여기서 복사본) → 고장 창 목록에 더함 → system.fit (랜덤포레스트 감지·원인)
[넣는 모양] 창(1초)마다 무작위로 하나 (주파수·세기·축 비율도 무작위)
    좁은 대역 잡음 : 5~235 Hz 중 한 곳, 폭 4~40 Hz        (베어링·마찰 같은 고장)
    순음 + 배음    : 3~200 Hz 중 하나 + 2·3배 성분          (기어·회전 부품)
    주기 충격      : 1초에 1~30번 '쿵', 15~220 Hz로 울림     (풀림·찍힘·이 빠짐)
    세기: 넣는 에너지 = 그 창 진동 에너지의 config.SYNTH_ENERGY 범위 배. 가속도 3축에만 넣음 (엔코더·전류는 그대로)
[지키는 것] 원래 데이터 파일과 원래 창은 바꾸지 않는다. 처음 5분(정상 기준)은 원래 창 그대로 두어 그 로봇 기준으로 비교된다.
           가상 이상 창은 synthetic=True로 표시되어 건강 점수 9점 눈금 맞추기에는 쓰지 않는다 (system.fit).
"""
import zlib

import numpy as np

import config as C

ACC = ("ax", "ay", "az")
LABEL_KIND = "unknown"


def _unit(x):
    x = x - x.mean()
    return x / max(float(x.std()), 1e-12)


def _band_noise(rng, n, fs):
    fc = rng.uniform(5, min(235.0, 0.49 * fs))
    bw = rng.uniform(4, 40)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[np.abs(f - fc) > bw / 2] = 0
    return _unit(np.fft.irfft(X, n))


def _tone(rng, n, fs):
    t = np.arange(n) / fs
    f0 = rng.uniform(3, min(200.0, 0.45 * fs))
    x = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    for h in (2, 3):
        if rng.random() < 0.6:
            x += rng.uniform(0.1, 0.6) * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 2 * np.pi))
    return _unit(x * (1 + rng.uniform(0, 0.5) * np.sin(2 * np.pi * rng.uniform(0.5, 5) * t)))


def _impulses(rng, n, fs):
    rate = rng.uniform(1, 30)
    fr = rng.uniform(15, min(220.0, 0.45 * fs))
    L = int(min(n, rng.uniform(0.02, 0.15) * fs))
    k = np.arange(L)
    ring = np.exp(-k / max(L / 4, 1)) * np.sin(2 * np.pi * fr * k / fs)
    hits = np.zeros(n)
    i = rng.uniform(0, fs / rate)
    while i < n:
        hits[int(i)] = rng.uniform(0.6, 1.4)
        i += (fs / rate) * rng.uniform(0.85, 1.15)
    return _unit(np.convolve(hits, ring)[:n])


SHAPES = (_band_noise, _tone, _impulses)


def inject(rec, rng):
    """1초 창 하나 → 가상 이상을 넣은 복사본 (라벨 = 처음 보는 이상)"""
    out = dict(rec)
    fs = float(rec["vib_fs"])
    n = min(len(rec[a]) for a in ACC if a in rec)
    shape = SHAPES[rng.integers(len(SHAPES))](rng, n, fs)
    lo, hi = C.SYNTH_ENERGY
    e = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))   # 넣는 에너지 ÷ 창 에너지
    w = rng.uniform(0.3, 1.0, len(ACC))
    w = w / w.max()
    for a, wa in zip(ACC, w):
        if a not in rec:
            continue
        x = np.asarray(rec[a], float)
        amp = np.sqrt(e * max(float(np.var(x)), 1e-10)) * wa
        y = x.copy()
        y[:n] += amp * shape
        out[a] = y.astype(np.float32)
    out.update(labeled=True, true_kind=LABEL_KIND, true_side="", true_sev=1.0, synthetic=True)
    return out


def synth_copy(recs, name):
    """정상 로봇 1대의 창들 → 복사본: 처음 BASELINE_PERIODS주기는 그대로(정상 기준), 그 뒤 창의 SYNTH_SHARE에 가상 이상.
    나머지 창은 복사본에서 뺀다 (원래 로봇에 이미 있으므로). 같은 로봇 이름이면 늘 같은 결과 (seed = 이름)."""
    rng = np.random.default_rng(C.SEED + zlib.crc32(name.encode("utf-8")))
    out = []
    for r in recs:
        if r["period"] < C.BASELINE_PERIODS:
            out.append(r)
        elif not r["bump"] and rng.random() < C.SYNTH_SHARE:
            out.append(inject(r, rng))
    return out
