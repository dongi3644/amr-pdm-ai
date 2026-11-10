"""[역할] 짧은 실측 기록들을 비교해 센서가 제대로 반응하는지 보는 점검 (AI 학습 아님)

    python 실측점검/check_recordings.py data/2026_11_10_test1.csv data/2026_11_10_182304.csv

학습에는 한 기록당 6분 이상과 각속도 3축이 필요하다. 그보다 짧거나 가속도만 있는 기록은 run.py로 학습할 수 없으므로,
여기서는 파이프라인과 같은 방법(단위 맞추기 → 1초 창 → 진동 특징)으로 기록끼리 숫자를 비교만 한다.
    - 기록 품질: 길이, 실제 샘플링, 빠진 구간, 중력(1 g) 확인
    - 진동 특징: 축마다 1초 창의 RMS·첨도·크레스트·저/중/고주파 에너지 (features.vib_features와 같은 계산)
    - 전류·전압: 평균, 범위
결과: 실측_결과/<날짜>/ 에 README.md(표)와 그래프
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config as C  # noqa: E402
from pipeline import dataio as D  # noqa: E402
from pipeline.features import vib_features  # noqa: E402
from pipeline import report  # noqa: E402,F401  (한글 글꼴·그래프 모양 설정)
import matplotlib.pyplot as plt  # noqa: E402

KO = dict(rms="진동 크기 RMS [mg]", kurt="첨도", crest="크레스트", low="저주파 1~20 Hz", mid="중주파 20~150 Hz", high="고주파 150 Hz~")


def load(path):
    """CSV → (표준 단위 표, 샘플링, 확인할 점). dataio와 같은 단위 맞추기를 씀"""
    data = D.Dataset()
    df = D._standardize(D.read_table(path), Path(path).name, data)
    t = df["time"].to_numpy(float)
    dt = np.diff(t)
    return df, 1 / np.median(dt), dt, data.warnings


def windows(df, fs):
    """1초 창마다 축별 진동 특징 (80%보다 샘플이 적은 창은 버림, dataio와 같은 기준)"""
    t = df["time"].to_numpy(float)
    idx = np.floor((t - t[0]) / C.WINDOW_S).astype(int)
    rows = []
    for w in np.unique(idx):
        part = df[idx == w]
        if len(part) < C.MIN_WINDOW_FILL * fs * C.WINDOW_S:
            continue
        row = {"창": int(w)}
        for ch in [c for c in C.VIB_CHANNELS if c in df.columns]:
            for k, v in vib_features(part[ch].to_numpy(float), fs).items():
                row[f"{ch}_{k}"] = v * (1000 if k == "rms" else 1)   # RMS는 mg로
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    out = PROJ / "실측_결과" / datetime.now().strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    info, feats, raw = [], {}, {}
    for p in paths:
        df, fs, dt, warns = load(p)
        name = p.stem
        lab = df["fault_kind"].iloc[0] if "fault_kind" in df.columns else ""
        spd = int(df["spd"].iloc[0]) if "spd" in df.columns else None
        a = np.sqrt((df[["ax", "ay", "az"]] ** 2).sum(axis=1))
        info.append({"파일": p.name, "라벨": lab, "속도 단계": spd, "길이(초)": round(float(df["time"].iloc[-1] - df["time"].iloc[0]), 1),
                     "샘플링(Hz)": round(fs, 1), "빠진 구간(간격 2배 넘음)": int((dt > 2 * np.median(dt)).sum()),
                     "가속도 크기 평균(g)": round(float(a.mean()), 3),
                     "각속도 열": "있음" if all(c in df.columns for c in ("gx", "gy", "gz")) else "없음",
                     "전압(V)": f"{df['voltage'].mean():.2f} ({df['voltage'].min():.2f}~{df['voltage'].max():.2f})" if "voltage" in df.columns else "없음(0)",
                     "전류(A)": f"{df['current'].mean():.3f} ({df['current'].min():.3f}~{df['current'].max():.3f})" if "current" in df.columns else "없음(0)",
                     "확인할 점": " / ".join(warns)})
        feats[name] = windows(df, fs)
        raw[name] = (df, fs, lab, spd)

    # 표: 축마다 특징의 평균 (1초 창들)
    axes = [c for c in C.VIB_CHANNELS if all(f"{c}_rms" in f.columns for f in feats.values())]
    lines = [f"# 실측 기록 점검 ({datetime.now():%Y-%m-%d %H:%M})", "",
             "> **AI 학습·성능 결과가 아닙니다.** 기록이 짧고(10초) 각속도가 없어 학습할 수 없으므로,",
             "> 파이프라인과 같은 방법으로 진동 특징을 계산해 **센서가 상태 변화에 반응하는지**만 비교했습니다.", "",
             "## 기록 품질", "", "| " + " | ".join(info[0].keys()) + " |", "|" + "---|" * len(info[0])]
    lines += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in info]
    lines += ["", "## 진동 특징 (1초 창 평균 ± 표준편차)", ""]
    names = list(feats)
    lines += ["| 축 | 특징 | " + " | ".join(names) + " | 배율 (" + f"{names[-1]} ÷ {names[0]}" + ") |", "|---|---|" + "---|" * (len(names) + 1)]
    for ch in axes:
        for k in KO:
            col = f"{ch}_{k}"
            vals = [feats[n][col] for n in names]
            cells = [f"{v.mean():.3g} ± {v.std():.2g}" for v in vals]
            ratio = vals[-1].mean() / vals[0].mean() if vals[0].mean() else float("nan")
            lines.append(f"| {ch} | {KO[k]} | " + " | ".join(cells) + f" | {ratio:.2g}배 |")
    lines += ["", f"1초 창 수: " + ", ".join(f"{n} {len(feats[n])}개" for n in names), ""]

    # 그래프: 시간 파형(az), 스펙트럼, 창별 RMS
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    for n, color in zip(names, [report.GRAY, report.BLUE, report.ORANGE, report.INK]):
        df, fs, lab, spd = raw[n]
        t = df["time"] - df["time"].iloc[0]
        ax[0].plot(t, df["az"], lw=0.6, color=color, label=f"{n} ({lab}, 속도 {spd})")
        x = df["az"].to_numpy(float) - df["az"].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 2
        ax[1].plot(np.fft.rfftfreq(len(x), 1 / fs), spec * 1000, lw=0.6, color=color, label=n)
        ax[2].plot(feats[n]["창"], feats[n]["az_rms"], "o-", color=color, label=n)
    ax[0].set(title="Z축 가속도 파형", xlabel="시간 [s]", ylabel="g")
    ax[1].set(title="Z축 가속도 주파수 성분", xlabel="주파수 [Hz]", ylabel="진폭 [mg]")
    ax[2].set(title="1초 창별 Z축 진동 크기", xlabel="창 (초)", ylabel="RMS [mg]")
    for a_ in ax:
        a_.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_기록비교.png", dpi=130)
    lines += ["![기록 비교](fig_기록비교.png)", ""]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("저장:", out)


if __name__ == "__main__":
    main()
