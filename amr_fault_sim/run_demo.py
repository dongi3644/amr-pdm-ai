"""데모 실행: 시뮬레이션 → 특징 추출 → 이상 감지 → 결과 저장·그래프

실행:  python run_demo.py
결과:  output/features.csv, output/result.png
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from simulator import AMRSimulator
from detector import extract, BaselineDetector

# 한글 폰트: 윈도우=맑은 고딕, 맥=애플고딕, 리눅스=Noto CJK(있으면 등록)
import glob
from matplotlib import font_manager
for _p in glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):
    font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "output"


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) 시뮬레이션
    records = AMRSimulator().run()
    print(f"창(window) 수: {len(records)}")

    # 2) 특징 추출
    rows = extract(records)

    # 3) 정상 기준 학습 (초기 BASELINE_LAPS 반복)
    det = BaselineDetector()
    det.fit([r for r in rows if r["lap"] < C.BASELINE_LAPS])
    for r in rows:
        r["health_score"] = det.score(r)
        r["risk"] = det.risk(r["health_score"]) if not r["bump"] else "-"

    # 4) CSV 저장
    keys = list(rows[0].keys())
    with open(f"{OUT}/features.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # 5) 반복(lap)별 요약 출력
    print("\nlap | 실제 고장(우측) | 평균 점수 | 대표 위험도")
    for lap in range(C.N_LAPS):
        sub = [r for r in rows if r["lap"] == lap and not r["bump"]]
        sc = np.mean([r["health_score"] for r in sub])
        print(f"{lap:3d} | {sub[0]['right_true']:.2f}          | {sc:6.2f}   | {det.risk(sc)}")

    # 6) 그래프
    idx = np.arange(len(rows))
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(idx, [r["right_true"] for r in rows], color="C3")
    ax[0].set_ylabel("실제 고장 정도\n(우측 바퀴)")
    ax[1].plot(idx, [r["left_band_low"] for r in rows], label="좌", alpha=.7)
    ax[1].plot(idx, [r["right_band_low"] for r in rows], label="우", alpha=.7)
    ax[1].set_ylabel("진동 저주파\n대역 에너지"); ax[1].legend(loc="upper left")
    ax[2].plot(idx, [r["left_temp"] for r in rows], label="좌")
    ax[2].plot(idx, [r["right_temp"] for r in rows], label="우")
    ax[2].set_ylabel("모터 온도 [°C]"); ax[2].legend(loc="upper left")
    # 문턱 통과 창은 점수 계산에서 제외 (순간 충격이라 고장과 무관)
    sc = np.array([np.nan if r["bump"] else r["health_score"] for r in rows])
    ax[3].plot(idx, sc, color="k", lw=.8)
    ax[3].axhline(3, ls="--", color="C1"); ax[3].axhline(6, ls="--", color="C3")
    ax[3].set_ylabel("이상 점수\n(3=주의, 6=경고)"); ax[3].set_xlabel("시간 (1초 창)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/result.png", dpi=110)
    print(f"\n저장: {OUT}/features.csv, {OUT}/result.png")


if __name__ == "__main__":
    main()
