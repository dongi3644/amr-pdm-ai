"""특징 추출: 1초 창의 원시 신호 → 숫자 특징

진동: RMS, 첨도, 크레스트, 대역 에너지(저·중·고), 회전 주파수(1X) 진폭
전류·온도: 평균
좌우 차이: 같은 바닥을 지나는 두 바퀴의 차이 → 노면 영향은 상쇄되고 한쪽 고장만 남음
"""
import numpy as np

import config as C

WHEELS = ("left", "right")
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high", "x1"]


def vib_features(x, fs, rot_hz):
    x = np.asarray(x, float) - np.mean(x)
    ms = np.mean(x ** 2) + 1e-12
    spec = np.abs(np.fft.rfft(x)) / len(x) * 2
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


def truth(rec):
    """정답 라벨: 가장 심한 고장의 종류·바퀴·정도 (사용자 데이터는 파일의 라벨을 그대로 씀)"""
    if "true_kind" in rec:
        return dict(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"])
    best = ("normal", "", 0.0)
    for w in WHEELS:
        for k in C.FAULT_KINDS:
            v = rec.get(f"{w}_true_{k}", 0.0)
            if v > best[2]:
                best = (k, w, v)
    return dict(true_kind=best[0] if best[2] > 0 else "normal", true_side=best[1], true_sev=best[2])


def extract(rec):
    rot_hz = rec["speed"] / C.ROBOT["wheel_radius"] / (2 * np.pi)
    row = dict(lap=rec["lap"], segment=rec["segment"], speed=rec["speed"], accel=rec["accel"], bump=rec["bump"])
    for w in WHEELS:
        for k, v in vib_features(rec[f"{w}_vib"], rec.get("vib_fs", C.SENSORS["vib_fs"]), rot_hz).items():
            row[f"{w}_{k}"] = v
        row[f"{w}_cur"] = float(np.mean(rec[f"{w}_cur"]))
        row[f"{w}_temp"] = float(rec[f"{w}_temp"])
    for k in VIB_KEYS + ["cur", "temp"]:
        row[f"diff_{k}"] = row[f"right_{k}"] - row[f"left_{k}"]
    row.update(truth(rec))
    return row


# AI 입력으로 쓰는 특징 (전류 잔차 res_* 는 models.CurrentModel이 추가)
BASE_FEATS = ([f"{w}_{k}" for w in WHEELS for k in VIB_KEYS + ["cur", "temp"]]
              + [f"diff_{k}" for k in VIB_KEYS + ["cur", "temp"]])
FEATS = BASE_FEATS + ["res_left", "res_right", "diff_res"]
