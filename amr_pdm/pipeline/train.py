"""학습 + 평가 + 보고서

1) AMRPipeline 학습: 전류 예측 → 이상 감지 4종(눈금 보정) → 원인 분류
2) 처음 보는 테스트 로봇 12대로 평가: 조기 감지, 오경보, 원인 정확도, 정지 위험 시점 예측 오차
3) 그래프 4장 + summary.md 저장
"""
import json

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score

import config as C
from pipeline import report
from pipeline.models import NORMAL_LABEL, kalman_forecast, label_ko, trend_forecast
from pipeline.system import AMRPipeline, per_robot_z


def records(df):
    rs = df.to_dict("records")
    for r in rs:
        r["bump"] = bool(r["bump"])
        r["true_side"] = r["true_side"] if isinstance(r["true_side"], str) else ""
    return rs


def evaluate_robot(pipe, det_key, rows):
    p = pipe.monitor(det_key, use_cause=(det_key == "ae"))
    causes = []
    for r in rows:
        o = p.update_row(dict(r))
        if o.get("cause_label"):
            causes.append((r["lap"], o["cause_label"]))
    p.finish()
    laps, sev = dict(p.lap_scores), {}
    for r in rows:
        sev[r["lap"]] = max(sev.get(r["lap"], 0), r["true_sev"])
    first = lambda st: next((lap for lap, s in p.events if s >= st), None)
    fail = next((l for l in sorted(sev) if sev[l] >= C.FAILURE_SEVERITY), None)
    onset = next((l for l in sorted(sev) if sev[l] > 0), None)
    return dict(lap_scores=laps, sev=sev, caution=first(1), warning=first(2), fail=fail, onset=onset,
                events=p.events, causes=causes, reason=p.reason)


def forecast_errors(e, fn):
    """첫 주의부터 9점 도달 직전까지 매 회차 예측해 실제 도달 회차와 비교 → [(오차, 90% 범위 안인가)]"""
    ls = e["lap_scores"]
    hit = next((l for l in sorted(ls) if ls[l] >= C.THRESH["danger"]), None)
    if hit is None or e["caution"] is None:
        return []
    out = []
    for lap in sorted(ls):
        if e["caution"] <= lap < hit:
            fc = fn([ls[l] for l in sorted(ls) if l <= lap])
            if fc:
                inside = fc["laps_lo"] <= hit - lap <= fc["laps_hi"] if "laps_lo" in fc else None
                out.append((abs(lap + fc["laps"] - hit), inside))
    return out


def main(force_data=False):
    from pipeline.dataset import build, scenarios
    C.REPORT_DIR.mkdir(exist_ok=True)
    print("[1/4] 데이터 생성 (가상 로봇 주행)", flush=True)
    D = {k: records(v) for k, v in build(force_data).items()}

    print("[2/4] 파이프라인 학습 (전류 예측 → 이상 감지 → 원인 분류)", flush=True)
    pipe = AMRPipeline().fit(D)
    pipe.save()
    dets, cause = pipe.bundle["detectors"], pipe.bundle["cause"]
    tn = [r for r in D["test"] if r["robot"].startswith("TN") and not r["bump"]]
    r2 = {w: float(r2_score([r[f"{w}_cur"] for r in tn], [r[f"pred_{w}"] for r in tn])) for w in ("left", "right")}

    print("[3/4] 평가 (처음 보는 테스트 로봇 12대)", flush=True)
    trows, Zt = per_robot_z(D["test"])
    sev_t = np.array([r["true_sev"] for r in trows])
    mk = (sev_t == 0) | (sev_t >= 0.3)
    auc = {k: float(roc_auc_score(sev_t[mk] >= 0.3, d.raw(Zt[mk]))) for k, d in dets.items()}
    me = (sev_t == 0) | ((sev_t > 0) & (sev_t < 0.3))
    auc_early = {k: float(roc_auc_score(sev_t[me] > 0, d.raw(Zt[me]))) for k, d in dets.items()}

    cm_mask = sev_t >= C.CAUSE_MIN_SEVERITY
    y_true = np.array([f"{r['true_kind']}|{r['true_side']}" for r in trows])[cm_mask]
    y_pred = np.array(cause.classes)[np.argmax(cause.proba(Zt[cm_mask]), 1)]
    cause_acc = float(np.mean(y_true == y_pred))
    kind_acc = float(np.mean([a.split("|")[0] == b.split("|")[0] for a, b in zip(y_true, y_pred)]))
    cm_labels = [c for c in cause.classes if c != NORMAL_LABEL] + [NORMAL_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    by_robot = {}
    for r in D["test"]:
        by_robot.setdefault(r["robot"], []).append(r)
    desc = {sc["robot"]: sc for sc in scenarios()["test"]}
    results = {key: {rid: evaluate_robot(pipe, key, rr) for rid, rr in by_robot.items()} for key in dets}
    robots_out = []
    for rid, rr in by_robot.items():
        e, sc = results["ae"][rid], desc[rid]
        item = dict(robot=rid, scenario=rr[0]["scenario"], onset=e["onset"], caution=e["caution"],
                    warning=e["warning"], fail=e["fail"])
        if e["caution"] is not None:
            item["sev_at_caution"] = round(e["sev"][e["caution"]], 2)
        if e["warning"] is not None and e["fail"] is not None:
            item["lead_laps"] = e["fail"] - e["warning"]
        hit = next((l for l in sorted(e["lap_scores"]) if e["lap_scores"][l] >= C.THRESH["danger"]), None)
        if hit is not None:
            item["danger_lap"], item["sev_at_danger"] = hit, round(e["sev"][hit], 2)   # 9점 도달 시 실제 고장 정도
        if e["causes"] and sc["faults"]:
            f, final = sc["faults"][0], e["causes"][-1][1]
            item["cause_pred"] = label_ko(final)
            item["cause_ok"] = final == f"{f['kind']}|{f['wheel']}"
            item["cause_reason"] = [[n, round(v, 3)] for n, v in e["reason"]]
        robots_out.append(item)

    def summarize(key):
        res = results[key]
        fr = [r for r in res if not r.startswith("TN")]
        nr = [r for r in res if r.startswith("TN")]
        leads = [res[r]["fail"] - res[r]["warning"] for r in fr if res[r]["warning"] is not None and res[r]["fail"] is not None]
        sev_c = [res[r]["sev"][res[r]["caution"]] for r in fr if res[r]["caution"] is not None]
        fa = sum(1 for r in nr if res[r]["caution"] is not None)
        det_ok = sum(1 for r in fr if res[r]["warning"] is not None and res[r]["fail"] is not None
                     and res[r]["warning"] < res[r]["fail"])
        same = sum(1 for r in fr if res[r]["caution"] is not None and res[r]["caution"] == res[r]["warning"])
        return dict(name=dets[key].name, auc=round(auc[key], 3), auc_early=round(auc_early[key], 3),
                    detected_before_fail=f"{det_ok}/{len(fr)}",
                    mean_lead_laps=round(float(np.mean(leads)), 1) if leads else None,
                    mean_sev_at_caution=round(float(np.mean(sev_c)), 2) if sev_c else None,
                    caution_equals_warning=f"{same}/{len(fr)}", false_alarm_robots=f"{fa}/{len(nr)}")

    def compare_forecast(fn):
        errs = [x for rid, e in results["ae"].items() for x in forecast_errors(e, fn)]
        inside = [i for _, i in errs if i is not None]
        return dict(mae=round(float(np.mean([a for a, _ in errs])), 2) if errs else None, n=len(errs),
                    coverage=round(float(np.mean(inside)), 2) if inside else None)

    comp = {k: summarize(k) for k in dets}
    metrics = dict(current_model_r2=r2, detector_comparison=comp,
                   cause_window_accuracy=round(cause_acc, 3), cause_kind_accuracy=round(kind_acc, 3),
                   cause_robot_accuracy=f"{sum(1 for r in robots_out if r.get('cause_ok'))}/"
                                        f"{sum(1 for r in robots_out if 'cause_ok' in r)}",
                   forecast_compare=dict(kalman=compare_forecast(kalman_forecast), trend=compare_forecast(trend_forecast)),
                   robots=robots_out,
                   data={k: dict(robots=len({r['robot'] for r in v}), windows=len(v)) for k, v in D.items()})
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] 그래프·보고서 저장", flush=True)
    report.fig_health_trend(results["ae"], {r["robot"]: r["scenario"] for r in robots_out}, C.REPORT_DIR / "fig_health_trend.png")
    report.fig_method_compare(comp, C.REPORT_DIR / "fig_method_compare.png")
    report.fig_confusion(cm, cm_labels, C.REPORT_DIR / "fig_confusion.png")
    report.fig_importance(cause.importance(), C.REPORT_DIR / "fig_importance.png")
    report.summary_md(metrics, C.REPORT_DIR / "summary.md")
    print(json.dumps({k: v for k, v in metrics.items() if k != "robots"}, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    import sys
    main(force_data="--regen" in sys.argv)
