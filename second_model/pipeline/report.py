"""평가 결과 → 그래프 + summary.md (실측 데이터 교차검증 결과)

[역할] workflow가 만든 숫자(metrics dict, 진단 표)를 사람이 보는 그림과 문서로 바꾼다. 계산은 하지 않는다.
[만드는 파일] (run.py train) reports/fig_auc.png      모델별 AUC 막대
                             reports/fig_units.png    평가 로봇별 시간에 따른 점수 (각자 자신이 빠진 모델로 진단)
                             reports/fig_confusion.png 원인 분류 혼동행렬
                             reports/fig_importance.png 원인 분류에 크게 쓰인 특징
                             reports/summary.md       위 수치를 표로 정리
              (run.py diagnose) reports/diagnose/<로봇>.png  로봇 1대 시간에 따른 건강 점수 (+ 전류, 배터리 전압)
"""
import glob
import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")                     # 창을 띄우지 않고 그림 파일로만 저장하는 방식 (서버·터미널에서도 동작)
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config as C
from pipeline.infer import feat_ko
from pipeline.models import label_ko

# 한글 글꼴 설정: 그래프의 한글이 네모(□)로 깨지지 않게, 컴퓨터에 있는 한글 글꼴을 찾아 쓴다
for _p in glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):   # 리눅스(Colab 등)의 한글 글꼴 등록
    font_manager.fontManager.addfont(_p)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)          # 글꼴을 못 찾을 때의 긴 경고는 숨김
_avail = {f.name for f in font_manager.fontManager.ttflist}                   # 이 컴퓨터에 설치된 글꼴 이름들
# 윈도우(맑은 고딕) → 맥(애플고딕) → 나눔고딕 → Noto 순서로 있는 것을 쓰고, 없으면 기본 글꼴
plt.rcParams["font.family"] = [f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"] if f in _avail] + ["DejaVu Sans"]
# 그래프 모양: 마이너스 기호 깨짐 방지, 위·오른쪽 테두리 없앰, 축·글자 색을 연하게
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titlesize": 11, "font.size": 10})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b5b4ad", "#0b0b0b"   # 그래프에 쓰는 색 (파랑=주 모델·정상, 주황=고장)


def fig_robot(table, title, note, path, calibrated=True):
    """로봇 1대 진단: 판정 주기별 건강 점수와 상태 기준선 (고장 데이터 없이 학습하면 9점 선은 그리지 않음)
    표에 전류·배터리 전압 칸이 있으면 그 아래에 같은 시간축으로 작은 그림을 하나씩 더 그린다 (축 하나에 하나씩)"""
    extra = [c for c in ("전류(A)", "배터리 전압(V)") if c in table and table[c].notna().any()]
    fig, axes = plt.subplots(1 + len(extra), 1, figsize=(10, 3.8 + 2.0 * len(extra)), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2] + [1] * len(extra)})
    ax = axes[0, 0]
    lines = [("주의", C.THRESH["caution"]), ("경고", C.THRESH["warning"])] + ([("정지 위험", C.THRESH["danger"])] if calibrated else [])
    for name, v in lines:                                  # 기준선(점선)과 이름표
        ax.axhline(v, color=GRAY, lw=1, ls="--")
        ax.text(table["경과(분)"].min(), v + 0.12, f"{name} {v:.0f}", color="#52514e", fontsize=9)
    ax.plot(table["경과(분)"], table["건강 점수"], color=BLUE, lw=2, marker="o", ms=3)   # 주기별 건강 점수
    ax.set(ylabel="진동 건강 점수", ylim=(0, max(C.THRESH["danger"] + 1.5, table["건강 점수"].max() * 1.05)))
    ax.set_title(f"{title} · {note}", loc="left", color=INK)
    for a, col in zip(axes[1:, 0], extra):                 # 전류 / 배터리 전압
        a.plot(table["경과(분)"], table[col], color=INK, lw=1.4, marker="o", ms=2)
        if col == "전류(A)" and "전류 상태" in table:      # '전류 높음'인 주기는 주황 점으로 표시
            hi = table["전류 상태"] == "높음"
            a.plot(table.loc[hi, "경과(분)"], table.loc[hi, col], "o", color=ORANGE, ms=5, label="평소보다 높음")
            if hi.any():
                a.legend(loc="upper left", fontsize=8, frameon=False)
        if col == "배터리 전압(V)":                        # 배터리 낮음 기준선
            a.axhline(C.BATTERY_LOW_V, color=GRAY, lw=1, ls="--")
            a.text(table["경과(분)"].min(), C.BATTERY_LOW_V, f" 배터리 낮음 {C.BATTERY_LOW_V:g} V", color="#52514e", fontsize=8, va="bottom")
        a.set_ylabel(col)
    axes[-1, 0].set_xlabel("경과 시간 (분)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_auc(m, path):
    """이상 감지 방법 비교 (로봇 단위 교차검증 AUC)"""
    items = sorted([v for v in m["detectors"].values() if v["auc"] is not None], key=lambda v: v["auc"])   # AUC 낮은 순 (막대가 아래부터 쌓임)
    if not items:                                          # 고장 창이 없어 AUC를 못 쟀으면 그림을 만들지 않음
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
    us = [u for u in m["unit_results"] if u["period_scores"]]   # 점수 기록이 있는 평가 로봇
    if not us:
        return False
    cols = min(4, len(us))                                 # 한 줄에 최대 4개
    rows = int(np.ceil(len(us) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows), sharey=True, squeeze=False)
    top = max(C.THRESH["danger"] + 1, max(s for u in us for _, s in u["period_scores"]) * 1.05)   # 모든 그림의 세로축을 같게
    for ax, u in zip(axes.flat, us):
        periods, scores = zip(*u["period_scores"])
        # x축: 주기 번호 → 분, 색: 고장 라벨 로봇은 주황, 정상은 파랑
        ax.plot([C.minutes(p + 1) for p in periods], scores, color=ORANGE if u["label"] not in (None, "normal|") else BLUE, lw=1.6)
        ax.axhline(C.THRESH["caution"], color=GRAY, lw=1, ls="--")
        ax.set_ylim(0, top)
        ax.set_title(f"{u['robot']} · {u['label_ko']}", fontsize=9, loc="left")
    for ax in list(axes.flat)[len(us):]:                   # 남는 빈칸은 숨김
        ax.axis("off")
    fig.supxlabel("경과 시간 (분)")
    fig.supylabel("건강 점수 (점선 3 = 주의)")
    fig.suptitle("평가 로봇별 건강 점수 (각 로봇은 자신이 빠진 모델로 진단) · 주황 = 고장 라벨, 파랑 = 정상", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def fig_confusion(m, path):
    """원인 분류 혼동행렬: 줄 = 실제 원인, 칸 = 예측한 원인, 숫자 = 창 수 (대각선에 몰릴수록 잘 맞춘 것)"""
    cm, labels = m["cause"]["confusion"], m["cause"]["labels"]
    if not cm:
        return False
    cm = np.array(cm)
    names = [label_ko(l) for l in labels]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 1.0 * len(names) + 2.5))   # 항목 수에 맞춰 그림 크기
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))   # 숫자가 클수록 진한 파랑
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):                           # 0이 아닌 칸에 숫자를 씀 (진한 칸은 흰 글자)
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
    """원인 분류에 크게 쓰인 특징 top개 막대그래프 (가장 큰 것이 맨 위)"""
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
    """교차검증 수치와 학습 메모를 reports/summary.md 문서로 (발표 자료에 옮겨 쓰기 쉬운 표 형식)"""
    p, u, f, c = m["protocol"], m["units"], m["forecast"], m["cause"]   # 평가 방식, 로봇 단위, 예측, 원인
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
              "## 원인 분류", "", f"- 창 단위 정확도 {c['window_accuracy']} · 로봇 단위 {c['unit_accuracy']}"
              + (f" (학습 겹에 같은 원인이 없어 맞힐 수 없던 고장 로봇 {c['unseen_units']}대는 정확도에서 뺌. 혼동행렬에는 포함)"
                 if c.get("unseen_units") else ""), "",
              "## 정지 위험 시점 예측", "", "| 방법 | 평균 오차(분) | 예측 횟수 |", "|---|---|---|",
              f"| 칼만 필터 | {f['kalman']['mae_min']} | {f['kalman']['n']} |", f"| 추세선 | {f['trend']['mae_min']} | {f['trend']['n']} |", "",
              f"칼만 필터 90% 범위 적중: {f['kalman']['coverage']}", "",
              f"## 건강 지표 품질 (진행형 고장 로봇): {m['health_indicator']}", "",
              "## 최종 모델 학습 메모", ""] + [f"- {n}" for n in notes]
    if m["skipped"]:
        lines += ["", "## 평가에서 빠진 항목", ""] + [f"- {s}" for s in m["skipped"]]
    path.write_text("\n".join(lines), encoding="utf-8")
