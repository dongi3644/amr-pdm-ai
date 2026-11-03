"""평가 결과 → 그래프 4장 + summary.md (모든 수치는 시뮬레이션 결과)"""
import glob
import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config as C
from pipeline.infer import feat_ko
from pipeline.models import label_ko

for _p in glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):
    font_manager.fontManager.addfont(_p)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_avail = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"]
                               if f in _avail] + ["DejaVu Sans"]
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titlesize": 11, "font.size": 10})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b5b4ad", "#0b0b0b"
NOTE = "시뮬레이션 데이터 결과 (실측 검증 전)"


def fig_health_trend(results, desc, path):
    """고장 로봇 6대: 회차별 건강 점수와 실제 고장 정도를 위아래로 나눠 표시 (이중 축 없음)"""
    fr = [r for r in results if not r.startswith("TN")]
    fig, ax = plt.subplots(2, len(fr), figsize=(16, 5), sharex=True, sharey="row",
                           gridspec_kw=dict(height_ratios=[3, 1]))
    for i, rid in enumerate(fr):
        e = results[rid]
        laps = sorted(e["lap_scores"])
        ax[0, i].plot(laps, [e["lap_scores"][l] for l in laps], color=BLUE, lw=2)
        ax[0, i].axhline(C.THRESH["caution"], color=GRAY, lw=1, ls="--")
        ax[0, i].axhline(C.THRESH["danger"], color=GRAY, lw=1)
        if e["caution"] is not None:
            ax[0, i].plot(e["caution"], e["lap_scores"][e["caution"]], "o", ms=8, color=ORANGE, mec="white", mew=1.5)
        ax[0, i].set_title(desc[rid].split(" (")[0])
        sev = sorted(e["sev"])
        ax[1, i].fill_between(sev, [e["sev"][l] for l in sev], color=GRAY, lw=0)
        ax[1, i].set_xlabel("회차")
    ax[0, 0].set_ylabel("건강 점수 (점선 3 주의, 실선 9 정지 위험)")
    ax[1, 0].set_ylabel("실제 고장 정도")
    fig.suptitle(f"고장이 진행될수록 건강 점수가 오른다 · 주황 점 = 첫 주의 · {NOTE}", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_method_compare(comp, path):
    """이상 감지 방법 비교: 값이 작을수록 좋은 '주의 시점 고장 정도' 하나로 비교"""
    items = [v for v in comp.values() if v["mean_sev_at_caution"] is not None]
    items = sorted(items, key=lambda v: -v["mean_sev_at_caution"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh([v["name"] for v in items], [v["mean_sev_at_caution"] for v in items], height=0.55,
                   color=[BLUE if "오토인코더" in v["name"] else GRAY for v in items])
    ax.bar_label(bars, fmt="%.2f", padding=4, color=INK)
    ax.set_xlabel("첫 주의 시점의 평균 고장 정도 (낮을수록 일찍 감지)")
    ax.set_title(f"이상 감지 방법 비교 · {NOTE}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_confusion(cm, labels, path):
    names = [label_ko(l) for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.spines[:].set_visible(False)
    ax.set_title(f"원인 분류 혼동행렬 (창 단위) · {NOTE}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_importance(importance, path, top=12):
    items = importance[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh([feat_ko(f) for f, _ in items], [v for _, v in items], height=0.6, color=BLUE)
    ax.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    ax.set_xlabel("특징 중요도")
    ax.set_title(f"원인 분류에 크게 쓰인 특징 · {NOTE}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_user(table, title, note, path, calibrated=True):
    """사용자 데이터 진단: 회차별 건강 점수 1개 선과 상태 기준선 (고장 데이터 없이 학습하면 9점 선은 그리지 않음)"""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lines = [("주의", C.THRESH["caution"]), ("경고", C.THRESH["warning"])] + ([("정지 위험", C.THRESH["danger"])] if calibrated else [])
    for name, v in lines:
        ax.axhline(v, color=GRAY, lw=1, ls="--")
        ax.text(table["회차"].min(), v + 0.12, f"{name} {v:.0f}", color="#52514e", fontsize=9)
    ax.plot(table["회차"], table["건강 점수"], color=BLUE, lw=2, marker="o", ms=3)
    ax.set(xlabel="회차", ylabel="건강 점수", ylim=(0, max(C.THRESH["danger"] + 1.5, table["건강 점수"].max() * 1.05)))
    ax.set_title(f"{title} · {note}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_md(m, path):
    lines = ["# AMR 고장예지 파이프라인 평가 요약", "",
             f"> {NOTE}. 발표 수치로 그대로 쓰지 말 것. 회차는 약 65초이며, 고장 진행 속도도 가정값이다.", "",
             "## 이상 감지", "",
             "| 방법 | AUC (0.3↑) | AUC (초기) | 정지 위험 전 경고 | 평균 여유(회차) | 주의 시점 고장 정도 | 오경보 로봇 |",
             "|---|---|---|---|---|---|---|"]
    for v in m["detector_comparison"].values():
        lines.append(f"| {v['name']} | {v['auc']} | {v['auc_early']} | {v['detected_before_fail']} | "
                     f"{v['mean_lead_laps']} | {v['mean_sev_at_caution']} | {v['false_alarm_robots']} |")
    f = m["forecast_compare"]
    lines += ["", "## 원인 분류", "",
              f"- 창 단위 정확도 {m['cause_window_accuracy']}, 고장 종류만 {m['cause_kind_accuracy']}, "
              f"로봇 단위 {m['cause_robot_accuracy']}", "",
              "## 정지 위험 시점 예측 (주의 이후 매 회차 예측의 오차)", "",
              "| 방법 | 평균 오차(회차) | 예측 횟수 |", "|---|---|---|",
              f"| 칼만 필터 | {f['kalman']['mae']} | {f['kalman']['n']} |",
              f"| 추세선 | {f['trend']['mae']} | {f['trend']['n']} |", "",
              f"칼만 필터 90% 범위에 실제 값이 든 비율: {f['kalman']['coverage']}", "",
              f"## 전류 예측 R²: {m['current_model_r2']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
