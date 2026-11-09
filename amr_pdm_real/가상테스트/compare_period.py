"""[역할] '1분 판정 점수를 어떻게 모을까'를 바꾸면 결과가 어떻게 달라지는지 비교한다 (코드 점검용, 성능 아님)

지금 방식: 1분 동안의 1초 점수들의 '중앙값'을 그 1분의 점수로 쓴다 → 튀는 창 몇 개에 흔들리지 않지만,
          1분 중 몇 초만 나타나는 간헐 고장은 묻힌다.
비교 방식: 높은 쪽 값(상위 25%·상위 10% 지점)을 쓰면 간헐 고장이 드러나지만, 정상일 때의 흔들림도 커진다.

    python 가상테스트/test_complex.py --model-out 모델경로     # 먼저 학습 (모델 저장)
    python 가상테스트/compare_period.py --model 모델경로       # 같은 시험 로봇으로 비교 (15분 안팎)

방법: 학습된 모델 하나로 시험 로봇마다 1초 점수를 한 번만 계산해 두고, 1분 점수 내는 방법만 바꿔 같은 상태 규칙
     (2번 연속이면 상태 변경, 9점 이상이면 바로)을 다시 적용한다. 모델·pipeline 코드는 바꾸지 않는다.
     남은 시간 예측·원인 분류는 이 비교에 넣지 않았다 (감지·오경보·경고 시점만 비교).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from pipeline import dataio as D  # noqa: E402
from pipeline.infer import Predictor, level  # noqa: E402
from pipeline.system import AMRPipeline  # noqa: E402
import sim_robots as S  # noqa: E402
from test_complex import make_specs  # noqa: E402

pct = lambda a, b: round(100.0 * a / b, 1) if b else None

# (이름, 1분 점수 내는 함수, 몇 번 연속이면 상태를 바꾸나)
VARIANTS = [
    ("중앙값 · 2번 연속 (지금 방식)", lambda h: float(np.median(h)), 2),
    ("상위 25% 지점 · 2번 연속", lambda h: float(np.quantile(h, 0.75)), 2),
    ("상위 10% 지점 · 2번 연속", lambda h: float(np.quantile(h, 0.90)), 2),
    ("상위 10% 지점 · 3번 연속", lambda h: float(np.quantile(h, 0.90)), 3),
]


def window_scores(bundle, recs):
    """로봇 1대를 처음부터 끝까지 넣어 주기마다 1초 점수 목록을 모음 → ([(주기, [점수...])], 실제 추론기의 상태 변경 이력)"""
    mon = Predictor(bundle, use_cause=False)            # 원인 분류는 끄고 (빠르게) 점수만
    got = []
    orig = mon._finish_period

    def capture(period):
        if mon.period_buf:
            got.append((period, list(mon.period_buf)))
        orig(period)

    mon._finish_period = capture
    for r in recs:
        mon.update(r)
    mon.finish()
    return got, list(mon.events)


def replay(periods, agg, need):
    """infer.Predictor._finish_period와 같은 상태 규칙을 1분 점수 함수 agg로 다시 적용 → (상태 변경 이력, 1분 점수들)"""
    status, pending, events, scores = 0, (0, 0), [], []
    for period, h in periods:
        s = agg(np.asarray(h))
        scores.append(s)
        lv = level(s)
        if lv != status:
            tgt, cnt = pending
            cnt = cnt + 1 if tgt == lv else 1
            pending = (lv, cnt)
            if cnt >= need or (lv > status and s >= C.THRESH["danger"]):
                status, pending = lv, (lv, 0)
                events.append((period, status))
        else:
            pending = (lv, 0)
    return events, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="test_complex.py --model-out 으로 저장한 모델")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default=str(PROJ / "가상데이터_결과값" / "복잡한고장_학습시험"))
    a = p.parse_args()
    out = Path(a.out)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    bundle = AMRPipeline.load(a.model).bundle
    C.PERIOD_S = bundle.get("period_s", C.PERIOD_S)
    _, test_specs = make_specs(a.seed, a.scale)

    robots = []                                            # 시험 로봇마다 (설정, 주기별 1초 점수, 주기별 고장 정도, 실제 이력)
    for i, s in enumerate(test_specs, 1):
        data = D.Dataset()
        recs, _ = D._windows(S.make_table(s), s["name"], data, None)
        periods, events = window_scores(bundle, recs)
        sev = {}
        for r in recs:
            sev[r["period"]] = max(sev.get(r["period"], 0.0), r["true_sev"])
        robots.append((s, periods, sev, events))
        if i % 20 == 0 or i == len(test_specs):
            log(f"1초 점수 계산 {i}/{len(test_specs)}")

    # 지금 방식 다시 적용 결과가 실제 추론기의 상태 변경 이력과 같은지 확인 (비교 방법이 맞는지 검사)
    same = sum(replay(pp, VARIANTS[0][1], VARIANTS[0][2])[0] == ev for _, pp, _, ev in robots)
    log(f"지금 방식 재현: {same}/{len(robots)}대 일치")

    rows, res = [], {}
    for name, agg, need in VARIANTS:
        for s, pp, sev, _ in robots:
            events, scores = replay(pp, agg, need)
            first = lambda st: next((q for q, x in events if x >= st), None)
            fail = next((q for q in sorted(sev) if sev[q] >= C.FAILURE_SEVERITY), None)
            onset = next((q for q in sorted(sev) if sev[q] > 0), None)
            c1, c2 = first(1), first(2)
            rows.append(dict(방식=name, 로봇=s["name"], 묶음=s["group"], 경우=s["cat"], 고장=s["fault"],
                             주의=c1 is not None, 감지까지=(C.minutes(c1 + 1 - onset) if (c1 is not None and onset is not None) else None),
                             정지위험=fail is not None,
                             미리경고=(fail is not None and c2 is not None and fail - c2 - 1 >= 0),
                             최고점수=max(scores) if scores else 0.0,
                             넘은주기=sum(x >= C.THRESH["caution"] for x in scores)))
    df = pd.DataFrame(rows)

    def stats(d):
        F, N = d[d["고장"]], d[~d["고장"]]
        fl = F[F["정지위험"]]
        return dict(robots=len(d), accuracy=pct(int(F["주의"].sum()) + int((~N["주의"]).sum()), len(d)),
                    detect=pct(int(F["주의"].sum()), len(F)), false_alarm=pct(int(N["주의"].sum()), len(N)),
                    warned=pct(int(fl["미리경고"].sum()), len(fl)),
                    delay=round(float(F["감지까지"].mean()), 1) if F["감지까지"].notna().any() else None,
                    normal_max=round(float(N["최고점수"].max()), 2) if len(N) else None,
                    normal_periods_over=int(N["넘은주기"].sum()) if len(N) else None)

    for name, _, _ in VARIANTS:
        d = df[df["방식"] == name]
        res[name] = dict(all=stats(d), groups={g: stats(d[d["묶음"] == g]) for g in dict.fromkeys(d["묶음"])},
                         cases={c: stats(d[d["경우"] == c]) for c in dict.fromkeys(d["경우"])})
    (out / "판정방식_비교.json").write_text(json.dumps(dict(reproduced=f"{same}/{len(robots)}", results=res),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res, same, len(robots), out / "판정방식_비교.md")
    print(json.dumps({n: r["all"] for n, r in res.items()}, ensure_ascii=False, indent=1))
    log(f"저장: {out / '판정방식_비교.md'}")


def write_md(res, same, n, path):
    f = lambda v, u="%": "-" if v is None else f"{v:g}{u}"
    names = list(res)
    L = ["# 1분 판정 점수 내는 방법 비교 (가상 로봇 · 코드 점검용 · 성능 아님)", "",
         "> 가상 로봇 결과라 실제 성능이 아닙니다. **어느 방법이 무엇을 얻고 무엇을 잃는지** 보는 자료입니다.", "",
         "- 같은 학습 모델(학습 160대), 같은 시험 로봇 170대. 1분 점수를 내는 방법과 연속 횟수만 바꿈",
         "- 지금 방식: 1분 동안 1초 점수들의 **중앙값**. 비교: 1초 점수 중 **상위 25%·상위 10% 지점** (높은 쪽 값)",
         f"- 검사: 지금 방식을 다시 적용한 결과가 실제 추론기 결과와 {same}/{n}대 일치",
         "- 남은 시간 예측·원인 분류는 비교하지 않음", "",
         "## 전체", "",
         "| 방법 | 판정 정답률 | 고장 감지율 | 오경보율 | 정지 위험 전 경고 | 고장 시작 → 주의 (1분 단위로 셈) | 정상 로봇 최고 점수 | 정상 로봇에서 3점 넘은 1분 |",
         "|---|---|---|---|---|---|---|---|"]
    for nm in names:
        v = res[nm]["all"]
        L.append(f"| {nm} | **{f(v['accuracy'])}** | {f(v['detect'])} | {f(v['false_alarm'])} | {f(v['warned'])} | "
                 f"{f(v['delay'], '분')} | {f(v['normal_max'], '점')} | {v['normal_periods_over']}번 |")
    L += ["", "## 경우별 고장 감지율 (정상은 오경보율)", "", "| 경우 | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for c in res[names[0]]["cases"]:
        cells = []
        for nm in names:
            v = res[nm]["cases"][c]
            cells.append(f"오경보 {f(v['false_alarm'])}" if v["detect"] is None else f(v["detect"]))
        L.append(f"| {c} ({res[names[0]]['cases'][c]['robots']}대) | " + " | ".join(cells) + " |")
    L += ["", "## 읽는 법", "",
          "- **정상 로봇 최고 점수**가 3점에 가까워질수록 실제 로봇에서는 오경보가 날 위험이 큽니다.",
          "  가상 정상 로봇은 실제보다 얌전하므로, 여기서 오경보 0%여도 실측에서는 다를 수 있습니다.",
          "- 높은 쪽 값을 쓰면 간헐 고장은 잘 잡히지만 정상일 때 점수도 올라갑니다. 연속 횟수를 늘리면 그만큼 오경보는 줄고 감지는 늦어집니다.",
          "- 방법을 바꿀지는 **실측 정상 기록**에서 점수가 얼마나 흔들리는지 보고 정해야 합니다 (`파라미터_설명.md`의 주의 2번)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
