"""[역할] 가상 성능 시험 결과(결과.json)를 누구나 한눈에 보게: 터미널 요약표 + 결과_요약.txt + 결과_그림.png (11/20)

    python 가상테스트/결과요약.py 가상테스트_결과/성능시험      # 실험 없이 지난 결과만 다시 보기
    (성능시험.py가 시험을 마치면 자동으로 부름. 성능시험.py --show 폴더 도 같음)
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

KIND = {"unbalance": "바퀴 불균형", "looseness": "체결 풀림", "bearing": "마찰 증가(베어링)", "gear": "기어 마모",
        "flat": "바퀴 찍힘", "unknown": "원인 미확정", None: "원인 미확정"}
UNSEEN = {"gear": "기어 마모", "flat": "바퀴 찍힘", "bearing_new": "처음 보는 주파수의 베어링"}
MAIN_KO = {"spec_rf": "주파수 모양 오토인코더 + 랜덤포레스트 (AI 2개)", "feat_rf": "특징 오토인코더 + 랜덤포레스트 (AI 2개)"}
OTHER_KO = {"ae_wheel": "특징 오토인코더 + 바퀴 점수", "rf": "랜덤포레스트만", "feat_rf": "특징 오토인코더 + 랜덤포레스트",
            "spec_ae": "주파수 모양 오토인코더만", "spec_wheel": "주파수 모양 오토인코더 + 바퀴 점수",
            "spec_rf": "주파수 모양 오토인코더 + 랜덤포레스트", "cnn_ae": "1D-CNN 오토인코더", "deep": "특징 AI + 1D-CNN",
            "deep_rf": "특징 AI + 1D-CNN + 랜덤포레스트"}


def width(s):
    """화면에서 차지하는 칸 수 (한글은 2칸)"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s, w):
    return s + " " * max(0, w - width(s))


def bar(pct, n=20):
    """0~100% → 막대 (█ 한 칸 = 5%)"""
    full = int(round(pct / 100 * n * 2))
    return "█" * (full // 2) + ("▌" if full % 2 else "") + "·" * (n - full // 2 - full % 2)


def grade(pct, lower_better=False):
    if pct is None:
        return ""
    if lower_better:
        return "아주 좋음" if pct <= 2 else "좋음" if pct <= 5 else "높음, 고쳐야 함"
    return "아주 좋음" if pct >= 95 else "좋음" if pct >= 90 else "더 고쳐야 함"


def counts(v):
    """결과 한 줄 → 대수 (맞힌 수, 전체 수). 예전 결과.json에는 대수가 없어 % × 대수로 되돌려 셈"""
    m = re.match(r"고장 (\d+)\(처음 보는 (\d+)\) · 정상 (\d+)", v["n"])
    nF, nU, nN = (int(x) for x in m.groups())
    nK = nF - nU
    r = lambda pct, n: int(round((pct or 0) * n / 100))
    kh, uh, fa = r(v["known"], nK), r(v["unseen"], nU), r(v["false_alarm"], nN)
    hits = kh + uh
    return dict(all=(hits + nN - fa, nF + nN), known=(kh, nK), unseen=(uh, nU), fa=(fa, nN), cause=(r(v["cause"], hits), hits))


def robot_list(items, cause=False):
    """놓친 · 원인을 틀린 로봇 → '바퀴 찍힘 2대 (세기 0.46 · 0.50배)' 같은 짧은 글"""
    groups = {}
    for w in items:
        name = UNSEEN.get(w.get("unseen")) or KIND.get(w.get("kind"), w.get("kind"))
        if cause:
            name = f"{name} → '{KIND.get(w.get('pred'), w.get('pred'))}'(으)로 봄"
        groups.setdefault(name, []).append(w)
    out = []
    for name, ws in groups.items():
        st = [f"{w['strength']:.2f}" for w in ws if "strength" in w]
        out.append(f"{name} {len(ws)}대" + (f" (세기 {' · '.join(st)}배)" if st and not cause else ""))
    return ", ".join(out)


def summary_text(res, lname, v):
    """한 조건의 결과 → 중학생도 읽을 수 있는 요약 글 (여러 줄)"""
    c = counts(v)
    total = c["all"][1]
    W = 24
    L = []
    L.append("═" * 74)
    L.append(f"  AMR 바퀴 고장 예지 AI · 시험 결과 (가상 로봇 {total}대, 실측 성능 아님)")
    L.append("═" * 74)
    cond = lname.split(" ", 1)[1] if " " in lname else lname
    L.append(f"  시험 조건: {cond}")
    L.append(f"  쓴 AI: {MAIN_KO.get(res.get('main'), res.get('main'))}")
    L.append("")
    rows = [("① 전체 정답률", v["accuracy"], c["all"], "대 중", "대 맞힘", False),
            ("② 아는 고장 찾기", v["known"], c["known"], "대 중", "대 찾음", False),
            ("③ 처음 보는 고장 찾기", v["unseen"], c["unseen"], "대 중", "대 찾음", False),
            ("④ 고장 원인 맞히기", v["cause"], c["cause"], "대(찾은 고장) 중", "대 맞힘", False),
            ("⑤ 잘못 울린 경보", v["false_alarm"], c["fa"], "대(정상 로봇) 중", "대", True)]
    for name, pct, (a, b), mid, tail, low in rows:
        if pct is None or b == 0:
            L.append(f"  {pad(name, W)}    –   (이 조건에는 해당 로봇 없음)")
            continue
        L.append(f"  {pad(name, W)}{pct:5.1f}%  {bar(pct)}  {pad(grade(pct, low), 14)} {b}{mid} {a}{tail}"
                 + ("  ← 낮을수록 좋음" if low else ""))
    L.append("")
    a, b = c["all"]
    L.append(f"  한 줄 요약: 로봇 100대를 검사하면 약 {round(100 * a / b)}대를 맞게 판정합니다."
             + (" 정상 로봇에 잘못 경보를 울린 적은 없습니다." if c["fa"][0] == 0 else f" 정상 로봇 100대 중 약 {round(100 * c['fa'][0] / c['fa'][1])}대에 잘못 경보가 울립니다."))
    L.append("")
    L.append("  [말 풀이]")
    L.append("   아는 고장      = AI가 배운 고장 (바퀴 불균형 · 체결 풀림 · 마찰 증가)")
    L.append("   처음 보는 고장 = AI가 한 번도 못 본 고장 (기어 마모 · 바퀴 찍힘 · 새 주파수 베어링) → '처음 보는 이상'으로 알리면 맞음")
    L.append("   잘못 울린 경보 = 멀쩡한 로봇인데 '고장'이라고 한 비율")
    if v.get("missed"):
        L.append(f"   놓친 고장 {len(v['missed'])}대: {robot_list(v['missed'])}")
    if v.get("wrong"):
        L.append(f"   원인을 틀린 {len(v['wrong'])}대: {robot_list(v['wrong'], cause=True)}")
    L.append("═" * 74)
    return "\n".join(L)


def compare_text(levels):
    """다른 모델들과 비교 (참고, 짧게)"""
    L = ["[참고 · 전문가용] 같은 시험에서 다른 모델 조합 (정답률 · 아는 고장 · 처음 보는 고장 · 잘못 울린 경보 · 원인, %)"]
    for lname, v in levels.items():
        L.append(f" {lname}")
        for k, x in v.items():
            f = lambda p: "  –  " if p is None else f"{p:5.1f}"
            L.append(f"   {pad(OTHER_KO.get(k, k), 36)} {f(x['accuracy'])} · {f(x['known'])} · {f(x['unseen'])} · {f(x['false_alarm'])} · {f(x['cause'])}")
    return "\n".join(L)


def chart(res, lname, v, path):
    """막대그림 1장: 높을수록 좋은 4개(파랑) + 낮을수록 좋은 잘못 울린 경보(회색), 95% 점선"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline import report  # noqa: F401  (한글 글꼴 설정)
        import matplotlib.pyplot as plt
    except Exception as e:                                   # 그림을 못 그려도 요약표는 그대로
        print(f"(그림은 건너뜀: {e})")
        return None
    c = counts(v)
    names = ["전체 정답률", "아는 고장 찾기", "처음 보는 고장 찾기", "고장 원인 맞히기", "잘못 울린 경보\n(낮을수록 좋음)"]
    vals = [v["accuracy"], v["known"], v["unseen"], v["cause"], v["false_alarm"]]
    cnt = [c["all"], c["known"], c["unseen"], c["cause"], c["fa"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = list(range(len(names)))[::-1]
    for yi, val, (a, b), i in zip(y, vals, cnt, range(5)):
        if val is None:
            continue
        col = "#2a6fdb" if i < 4 else "#8a8f98"
        ax.barh(yi, val, height=0.56, color=col)
        inside = val >= 40                                   # 막대가 길면 막대 안 왼쪽에 흰 글씨, 짧으면 막대 오른쪽에 검은 글씨
        ax.text(2 if inside else val + 1.2, yi, f"{val:.1f}%   ({b}대 중 {a}대)", va="center", ha="left",
                color="white" if inside else "#1d1d1f", fontsize=12, fontweight="bold" if inside else "normal")
    ax.axvline(95, color="#1d1d1f", ls="--", lw=1, zorder=0.5)
    ax.text(95, len(names) - 0.35, "95%", ha="center", fontsize=9, color="#1d1d1f")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(axis="x", color="#e3e5e8")
    ax.set_axisbelow(True)
    ax.set_title(f"AMR 바퀴 고장 예지 AI 시험 결과 (가상 로봇 {c['all'][1]}대)", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def show(folder):
    """결과 폴더 → 터미널 요약 + 결과_요약.txt + 결과_그림.png"""
    folder = Path(folder)
    res = json.loads((folder / "결과.json").read_text(encoding="utf-8"))
    main = res.get("main", "spec_rf")
    texts = []
    for i, (lname, lv) in enumerate(res["levels"].items()):
        if main not in lv:
            continue
        texts.append(summary_text(res, lname, lv[main]))
        png = chart(res, lname, lv[main], folder / ("결과_그림.png" if len(res["levels"]) == 1 else f"결과_그림_{i + 1}.png"))
        if png:
            texts.append(f"  그림: {png}")
    texts.append(compare_text(res["levels"]))                # 전문가용 비교는 맨 뒤에 짧게
    text = "\n\n".join(texts)
    print(text)
    (folder / "결과_요약.txt").write_text(text + "\n", encoding="utf-8")
    print(f"  요약 글: {folder / '결과_요약.txt'}")


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "가상테스트_결과" / "성능시험")
