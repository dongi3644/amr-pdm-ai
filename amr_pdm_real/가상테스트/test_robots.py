"""[역할] 무작위 가상 로봇 N대(기본 100대)로 '처음 보는 로봇' 시험을 하고 결과를 %로 정리한다 (코드 점검용, 성능 아님)

    python 가상테스트/test_robots.py                  # 시험 로봇 100대, 결과는 가상데이터_결과값/로봇100대/
    python 가상테스트/test_robots.py --n 200 --seed 7

순서
    1) 학습용 가상 로봇: 정상 12대 + 고장 종류(3) × 바퀴(4)마다 2대 = 36대
    2) 시험용 가상 로봇 N대: 한 대씩 동전을 던져(반반) 정상 또는 고장. 고장이면 종류·바퀴·진행 속도·최종 고장 정도를 무작위로
    3) 학습용 로봇으로만 모델을 학습하고, 시험 로봇은 학습에 전혀 쓰지 않은 채 처음부터 끝까지 진단
       (평가는 workflow.cross_validate를 '겹 1개 = 시험 로봇 전체'로 불러, 실측 교차검증과 똑같은 채점 코드를 씀)
    4) 로봇 단위·창 단위 결과를 %로 → README.md, metrics.json, 로봇별_결과.csv, 그래프

주의: 신호와 고장을 사람이 식으로 만들었으므로 이 %는 'AI가 실제 로봇 고장을 이만큼 맞힌다'는 뜻이 아니다.
가상 CSV는 data/에 쓰지 않고 메모리에서만 쓴다. 프로젝트의 artifacts/·reports/도 건드리지 않는다.
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
from pipeline import dataio as D, report, workflow as W  # noqa: E402
from pipeline.models import NORMAL_LABEL, label_ko  # noqa: E402
import sim_robots as S  # noqa: E402

pct = lambda a, b: round(100.0 * a / b, 1) if b else None   # a/b → % (분모가 0이면 None)


def add_robot(data, spec):
    """가상 로봇 1대의 표 → 1초 창으로 잘라 Dataset에 넣음 (dataio.load가 CSV 한 파일을 읽을 때와 같은 함수)"""
    recs, fs = D._windows(S.make_table(spec), spec["name"], data, None)
    data.robots[spec["name"]], data.fs[spec["name"]] = recs, fs


def main():
    p = argparse.ArgumentParser(description="무작위 가상 로봇 시험 (코드 점검용)")
    p.add_argument("--n", type=int, default=100, help="시험 로봇 수")
    p.add_argument("--seed", type=int, default=2026, help="같은 값이면 같은 로봇들이 만들어짐")
    p.add_argument("--fault-ratio", type=float, default=0.5, help="시험 로봇 중 고장 로봇이 될 확률")
    p.add_argument("--out", default=str(PROJ / "가상데이터_결과값" / "로봇100대"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}초] {s}", flush=True)

    # 1) 학습용 로봇 36대 (고장 종류·바퀴 12가지를 2대씩 넣어 원인 모델이 모든 원인을 배우게 함)
    train_specs = [S.random_spec(rng, f"학습_정상{i + 1:02d}", False) for i in range(12)]
    train_specs += [S.random_spec(rng, f"학습_{k}_{w}_{j + 1}", True, k, w) for k in S.KINDS for w in S.WHEELS for j in range(2)]
    # 2) 시험용 로봇 N대 (정상/고장, 고장 종류·바퀴 모두 무작위)
    test_specs = [S.random_spec(rng, f"시험{i + 1:03d}", rng.random() < a.fault_ratio) for i in range(a.n)]

    data = D.Dataset()
    for i, spec in enumerate(train_specs + test_specs, 1):
        add_robot(data, spec)
        if i % 20 == 0 or i == len(train_specs) + len(test_specs):
            log(f"가상 로봇 만드는 중 {i}/{len(train_specs) + len(test_specs)}")
    test_names = [s["name"] for s in test_specs]

    # 3) 학습(36대) → 시험(N대) : 겹 하나짜리 '교차검증'으로 부르면 채점 코드가 실측 평가와 같아진다
    W.make_folds = lambda data, names: [test_names]
    log("학습 + 시험 로봇 진단 시작 (시간이 걸립니다)")
    m = W.cross_validate(data, "per_unit", progress=log)
    log("진단 끝, 결과 정리")

    # 4) 로봇별 결과 표
    ko2raw = {label_ko(f"{k}|{w}"): f"{k}|{w}" for k in S.KINDS for w in S.WHEELS}
    ko2raw[label_ko(NORMAL_LABEL)] = NORMAL_LABEL
    by_name = {u["robot"]: u for u in m["unit_results"]}
    rows = []
    for s in test_specs:
        u = by_name.get(s["name"])
        if u is None:
            continue
        pred = ko2raw.get(u["cause_pred"]) if u["cause_pred"] else None
        alarm = u["caution"] is not None
        rows.append({
            "로봇": s["name"], "실제": "고장" if s["fault"] else "정상",
            "고장 종류": C.KIND_KO[s["kind"]] if s["fault"] else "", "고장 바퀴": C.WHEEL_KO.get(s["wheel"], "") if s["fault"] else "",
            "최종 고장 정도": round(s["final"], 2) if s["fault"] else 0.0,
            "고장 시작(분)": round(s["onset"], 1) if s["fault"] else None, "기록(분)": round(s["minutes"], 1),
            "최고 점수": u["max_score"],
            "첫 주의(분)": C.minutes(u["caution"] + 1) if alarm else None,
            "첫 경고(분)": C.minutes(u["warning"] + 1) if u["warning"] is not None else None,
            "판정": ("맞음 (고장 감지)" if alarm else "놓침") if s["fault"] else ("오경보" if alarm else "맞음 (정상)"),
            "추정 원인": u["cause_pred"] or "",
            "종류 맞음": (pred.split("|")[0] == s["kind"]) if (s["fault"] and pred and pred != NORMAL_LABEL) else None,
            "종류+바퀴 맞음": (pred == f"{s['kind']}|{s['wheel']}") if (s["fault"] and pred and pred != NORMAL_LABEL) else None,
            "경고 여유(분)": C.minutes(u["lead_periods"]) if u["lead_periods"] is not None and u["lead_periods"] >= 0 else None,
            "정지 위험까지 감": u["fail"] is not None})
    df = pd.DataFrame(rows)
    df.to_csv(out / "로봇별_결과.csv", index=False, encoding="utf-8-sig")

    # 5) % 정리
    F, N = df[df["실제"] == "고장"], df[df["실제"] == "정상"]
    hit_f, ok_n = int(F["첫 주의(분)"].notna().sum()), int(N["첫 주의(분)"].isna().sum())
    fail_f = F[F["정지 위험까지 감"]]
    warned = int(fail_f["경고 여유(분)"].notna().sum())
    caused = F[F["종류 맞음"].notna()]
    res = dict(
        robots=dict(test=len(df), fault=len(F), normal=len(N), train=len(train_specs)),
        robot_accuracy_pct=pct(hit_f + ok_n, len(df)),
        fault_detect_pct=pct(hit_f, len(F)), normal_ok_pct=pct(ok_n, len(N)), false_alarm_pct=pct(len(N) - ok_n, len(N)),
        warned_before_fail_pct=pct(warned, len(fail_f)), fail_robots=len(fail_f),
        mean_lead_min=round(float(fail_f["경고 여유(분)"].mean()), 1) if warned else None,
        cause_kind_pct=pct(int(caused["종류 맞음"].sum()), len(caused)),
        cause_kind_wheel_pct=pct(int(caused["종류+바퀴 맞음"].sum()), len(caused)), cause_robots=len(caused),
        by_kind={C.KIND_KO[k]: dict(robots=int((F["고장 종류"] == C.KIND_KO[k]).sum()),
                                    detect_pct=pct(int(F.loc[F["고장 종류"] == C.KIND_KO[k], "첫 주의(분)"].notna().sum()),
                                                   int((F["고장 종류"] == C.KIND_KO[k]).sum())),
                                    kind_ok_pct=pct(int(caused.loc[caused["고장 종류"] == C.KIND_KO[k], "종류 맞음"].sum()),
                                                    int((caused["고장 종류"] == C.KIND_KO[k]).sum())))
                 for k in S.KINDS},
        mild=dict(robots=int((F["최종 고장 정도"] < C.FAILURE_SEVERITY).sum()),
                  detect_pct=pct(int(F.loc[F["최종 고장 정도"] < C.FAILURE_SEVERITY, "첫 주의(분)"].notna().sum()),
                                 int((F["최종 고장 정도"] < C.FAILURE_SEVERITY).sum()))),
        windows={v["name"]: dict(auc=v["auc"], detect_pct=None if v["recall"] is None else round(100 * v["recall"], 1),
                                 false_alarm_pct=None if v["false_positive"] is None else round(100 * v["false_positive"], 1))
                 for v in m["detectors"].values()},
        forecast=m["forecast"], skipped=m["skipped"], seed=a.seed, seconds=round(time.time() - t0))
    (out / "metrics.json").write_text(json.dumps(dict(summary=res, cv_metrics=m), ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    report.fig_auc(m, out / "fig_auc.png")
    report.fig_confusion(m, out / "fig_confusion.png")
    write_md(res, df, out / "README.md", a)
    print(json.dumps({k: v for k, v in res.items() if k not in ("forecast",)}, ensure_ascii=False, indent=1))
    log(f"저장: {out}")


def write_md(r, df, path, a):
    """결과 % → README.md"""
    f = lambda v, unit="%": "-" if v is None else f"{v:g}{unit}"
    k = r["forecast"]["kalman"]
    lines = [
        f"# 가상 로봇 {r['robots']['test']}대 시험 결과 (코드 점검용 · 성능 아님)", "",
        "> **이 %는 발표·성능 자료로 쓰지 않습니다.** 진동 신호와 고장을 사람이 식으로 만든 가상 로봇이라,",
        "> 'AI가 실제 로봇 고장을 이만큼 맞힌다'는 뜻이 아닙니다. 실제 성능은 실측 CSV로 `python run.py train`을 돌려 잽니다.", "",
        f"- 만든 도구: `가상테스트/test_robots.py --n {a.n} --seed {a.seed}` (같은 명령이면 같은 결과)",
        f"- 학습 로봇 {r['robots']['train']}대 (정상 12 + 고장 종류 3 × 바퀴 4 × 2대) → 시험 로봇은 학습에 전혀 쓰지 않음",
        f"- 시험 로봇 {r['robots']['test']}대: 무작위로 정상 {r['robots']['normal']}대 · 고장 {r['robots']['fault']}대",
        "  - 고장 로봇은 종류·바퀴·고장 시작(6~8분)·진행 시간(6~12분)·최종 고장 정도(0.5~1.0)·진동에 드러나는 세기를 무작위로",
        "  - 정상 로봇은 기록 10~16분, 로봇마다 진동 크기·잡음·속도·배터리 전압이 다름",
        f"- 설정: 판정 주기 {C.PERIOD_S:g}초 · 정상 기준 per_unit · 주의 3점 / 경고 6점 / 위험 9점 · 걸린 시간 {r['seconds'] // 60}분", "",
        "## 로봇 단위 결과", "",
        "| 항목 | 결과 |", "|---|---|",
        f"| **전체 판정 정답률** (고장은 감지, 정상은 경보 없음) | **{f(r['robot_accuracy_pct'])}** |",
        f"| 고장 로봇 감지율 ('주의' 이상이 한 번이라도 뜸) | {f(r['fault_detect_pct'])} ({r['robots']['fault']}대 중) |",
        f"| 정상 로봇을 정상으로 판정 | {f(r['normal_ok_pct'])} ({r['robots']['normal']}대 중) |",
        f"| 정상 로봇 오경보율 | {f(r['false_alarm_pct'])} |",
        f"| 정지 위험(고장 정도 0.8) 전에 '경고' | {f(r['warned_before_fail_pct'])} ({r['fail_robots']}대 중) |",
        f"| 경고 → 정지 위험 평균 여유 | {f(r['mean_lead_min'], '분')} |",
        f"| 원인: 고장 종류 정확도 | {f(r['cause_kind_pct'])} (원인을 낸 {r['cause_robots']}대 중) |",
        f"| 원인: 종류 + 바퀴까지 정확도 | {f(r['cause_kind_wheel_pct'])} (바퀴 위치는 단순화한 가정이라 참고만) |",
        f"| 가벼운 고장(최종 정도 0.8 미만) 감지율 | {f(r['mild']['detect_pct'])} ({r['mild']['robots']}대 중) |",
        f"| 남은 시간 예측 평균 오차 (칼만 필터) | {f(k['mae_min'], '분')} ({k['n']}회) · 90% 범위 적중 {f(None if k['coverage'] is None else round(100 * k['coverage']))} |",
        "", "### 고장 종류별", "", "| 고장 종류 | 시험 로봇 | 감지율 | 종류 정확도 |", "|---|---|---|---|"]
    lines += [f"| {name} | {v['robots']}대 | {f(v['detect_pct'])} | {f(v['kind_ok_pct'])} |" for name, v in r["by_kind"].items()]
    lines += ["", "## 1초 창 단위 (모델 5개 비교, 같은 창)", "",
              "| 이상 감지 방법 | AUC | 고장 창 감지율 (3점 이상) | 정상 창 오경보율 |", "|---|---|---|---|"]
    lines += [f"| {name} | {v['auc']} | {f(v['detect_pct'])} | {f(v['false_alarm_pct'])} |" for name, v in r["windows"].items()]
    miss = df[df["판정"].isin(["놓침", "오경보"])]
    lines += ["", "## 틀린 로봇", ""]
    if len(miss):
        lines += ["| 로봇 | 판정 | 실제 | 최종 고장 정도 | 최고 점수 |", "|---|---|---|---|---|"]
        lines += [f"| {x['로봇']} | {x['판정']} | {x['실제']} {x['고장 바퀴']} {x['고장 종류']} | {x['최종 고장 정도']} | {x['최고 점수']} |"
                  for _, x in miss.iterrows()]
    else:
        lines.append("없음")
    if r["skipped"]:
        lines += ["", "평가에서 빠진 로봇: " + "; ".join(r["skipped"])]
    lines += ["", "파일: `로봇별_결과.csv` (100대 전부), `metrics.json` (모든 수치), `fig_auc.png`, `fig_confusion.png`", "",
              "## 이 결과로 알 수 있는 것 / 없는 것", "",
              "- 알 수 있는 것: 처음 보는 로봇 100대를 넣어도 진단·채점 코드가 오류 없이 돌고, 결과가 % 형식으로 정리된다.",
              "- 알 수 없는 것: 실제 로봇에서의 정확도. 가상 고장 신호는 식으로 만든 것이라 실제 고장보다 깨끗하고 일정하다.",
              "- 실제로는 로봇(차체)이 1대뿐이라 이렇게 100대를 시험할 수 없다. 실측 성능은 같은 차체의 다른 기록으로 잰 교차검증 수치다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
