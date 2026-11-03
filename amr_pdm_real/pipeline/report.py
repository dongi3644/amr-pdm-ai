"""평가 결과 → 그래프 + summary.md (실측 데이터 교차검증 결과)

[역할] workflow가 만든 숫자(metrics dict, 진단 표)를 사람이 보는 그림과 문서로 바꾼다. 계산은 하지 않는다.
[만드는 파일] (run.py train) reports/fig_auc.png      모델별 AUC 막대
                             reports/fig_units.png    평가 로봇별 시간에 따른 점수 (각자 자신이 빠진 모델로 진단)
                             reports/fig_confusion.png 원인 분류 혼동행렬
                             reports/fig_importance.png 원인 분류에 크게 쓰인 특징
                             reports/summary.md       위 수치를 표로 정리
              (run.py diagnose) reports/diagnose/<로봇>.png  로봇 1대 시간에 따른 점수
"""
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
plt.rcParams["font.family"] = [f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"] if f in _avail] + ["DejaVu Sans"]
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titlesize": 11, "font.size": 10})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b5b4ad", "#0b0b0b"


def fig_robot(table, title, note, path, calibrated=True):
    """로봇 1대 진단: 판정 주기별 건강 점수와 상태 기준선 (고장 데이터 없이 학습하면 9점 선은 그리지 않음)"""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lines = [("주의", C.THRESH["caution"]), ("경고", C.THRESH["warning"])] + ([("정지 위험", C.THRESH["danger"])] if calibrated else [])
    for name, v in lines:
        ax.axhline(v, color=GRAY, lw=1, ls="--")
        ax.text(table["경과(분)"].min(), v + 0.12, f"{name} {v:.0f}", color="#52514e", fontsize=9)
    ax.plot(table["경과(분)"], table["건강 점수"], color=BLUE, lw=2, marker="o", ms=3)
    ax.set(xlabel="경과 시간 (분)", ylabel="건강 점수", ylim=(0, max(C.THRESH["danger"] + 1.5, table["건강 점수"].max() * 1.05)))
    ax.set_title(f"{title} · {note}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_auc(m, path):
    """이상 감지 방법 비교 (로봇 단위 교차검증 AUC)"""
    items = sorted([v for v in m["detectors"].values() if v["auc"] is not None], key=lambda v: v["auc"])
    if not items:
        return False
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh([v["name"] for v in items], [v["auc"] for v in items], height=0.55,
                   color=[BLUE if "오토인코더" in v["name"] else GRAY for v in items])
    ax.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("AUC (고장 창 vs 정상 창, 높을수록 좋음)")
    ax.set_title(f"이상 감지 방법 비교 · 로봇 단위 교차검증 {m['protocol']['folds']}겹", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_units(m, path):
    """평가 로봇별 시간에 따른 건강 점수 (작은 그림 여러 개, 같은 눈금)"""
    us = [u for u in m["unit_results"] if u["period_scores"]]
    if not us:
        return False
    cols = min(4, len(us))
    rows = int(np.ceil(len(us) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows), sharey=True, squeeze=False)
    top = max(C.THRESH["danger"] + 1, max(s for u in us for _, s in u["period_scores"]) * 1.05)
    for ax, u in zip(axes.flat, us):
        periods, scores = zip(*u["period_scores"])
        ax.plot([C.minutes(p + 1) for p in periods], scores, color=ORANGE if u["label"] not in (None, "normal|") else BLUE, lw=1.6)
        ax.axhline(C.THRESH["caution"], color=GRAY, lw=1, ls="--")
        ax.set_ylim(0, top)
        ax.set_title(f"{u['robot']} · {u['label_ko']}", fontsize=9, loc="left")
    for ax in list(axes.flat)[len(us):]:
        ax.axis("off")
    fig.supxlabel("경과 시간 (분)")
    fig.supylabel("건강 점수 (점선 3 = 주의)")
    fig.suptitle("평가 로봇별 건강 점수 (각 로봇은 자신이 빠진 모델로 진단) · 주황 = 고장 라벨, 파랑 = 정상", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def fig_confusion(m, path):
    cm, labels = m["cause"]["confusion"], m["cause"]["labels"]
    if not cm:
        return False
    cm = np.array(cm)
    names = [label_ko(l) for l in labels]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 1.0 * len(names) + 2.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set(xlabel="예측", ylabel="실제")
    ax.spines[:].set_visible(False)
    ax.set_title("원인 분류 혼동행렬 (창 단위, 교차검증)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_importance(importance, path, top=12):
    items = importance[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh([feat_ko(f) for f, _ in items], [v for _, v in items], height=0.6, color=BLUE)
    ax.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    ax.set_xlabel("특징 중요도")
    ax.set_title("원인 분류에 크게 쓰인 특징 (최종 모델)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_md(m, notes, path):
    p, u, f, c = m["protocol"], m["units"], m["forecast"], m["cause"]
    lines = ["# AMR 고장예지 파이프라인 평가 요약 (실측 데이터)", "",
             f"- 평가 방식: {p['split']} 교차검증 {p['folds']}겹, 정상 기준 {p['baseline']}, 판정 주기 {p['period_s']:g}초. {p['note']}",
             f"- 평가 창: {m['windows']['n']:,}개 (고장 {m['windows']['fault']:,}개)", "",
             "## 이상 감지", "", "| 방법 | AUC | 고장 창 감지율 (3점 이상) | 정상 창 오경보율 |", "|---|---|---|---|"]
    for v in m["detectors"].values():
        lines.append(f"| {v['name']} | {v['auc']} | {v['recall']} | {v['false_positive']} |")
    lines += ["", "## 로봇 단위", "",
              f"- 고장 로봇 감지: {u['fault_detected']} · 정상 로봇 오경보: {u['false_alarm']}"
              + (f" (그중 라벨 없이 정상으로 간주한 로봇 {u['normal_assumed']}대)" if u["normal_assumed"] else ""),
              f"- 정지 위험 전 경고: {u['warned_before_fail']} · 평균 여유: {u['mean_lead_min']}분", "",
              "## 원인 분류", "", f"- 창 단위 정확도 {c['window_accuracy']} · 로봇 단위 {c['unit_accuracy']}", "",
              "## 정지 위험 시점 예측", "", "| 방법 | 평균 오차(분) | 예측 횟수 |", "|---|---|---|",
              f"| 칼만 필터 | {f['kalman']['mae_min']} | {f['kalman']['n']} |", f"| 추세선 | {f['trend']['mae_min']} | {f['trend']['n']} |", "",
              f"칼만 필터 90% 범위 적중: {f['kalman']['coverage']}", "",
              f"## 건강 지표 품질 (진행형 고장 로봇): {m['health_indicator']}", "",
              "## 배터리 소모 이상", "",
              f"- 정상 로봇 오경보: {m['battery']['normal_alarm']} · 배터리 라벨 로봇 감지: {m['battery']['battery_fault_detected']}"
              f" · 다른 고장 로봇에서 함께 경고: {m['battery']['other_fault_alarm']}", "",
              "## 최종 모델 학습 메모", ""] + [f"- {n}" for n in notes]
    if m["skipped"]:
        lines += ["", "## 평가에서 빠진 항목", ""] + [f"- {s}" for s in m["skipped"]]
    path.write_text("\n".join(lines), encoding="utf-8")
