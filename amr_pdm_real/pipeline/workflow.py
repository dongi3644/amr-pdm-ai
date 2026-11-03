"""학습 · 진단 · 평가 흐름 (실측 데이터 전용)

[역할] '어떤 로봇으로 학습하고 어떤 로봇으로 평가하나'를 정하는 절차 모음. 모델 계산은 system·infer에 맡긴다.
[흐름에서 위치] run.py → (여기) → system.AMRPipeline.fit / infer.Predictor → report.py

[함수]
    train()           로봇들을 정상/고장으로 나눠 파이프라인 학습 + 칼만 잡음 조정
    tune_kalman()     학습 고장 로봇의 주기별 점수로 칼만 필터 잡음 값을 격자 탐색
    diagnose()        로봇 1대를 처음부터 끝까지 진단해 판정 주기(기본 1분)별 표로
    make_folds()      교차검증 겹 나누기 (로봇 단위)
    cross_validate()  겹마다: 평가 로봇을 뺀 학습 → 평가 로봇 진단 → 창·로봇 단위 결과 수집
    summarize()       모은 결과 → AUC·감지율·오경보·원인 정확도·예측 오차·건강 지표 품질
    hi_metrics()      건강 지표 품질 (단조성·추세 일치도·고장 시점 일관성)

    pipe, notes = train(data)                 # 모든 로봇으로 최종 모델 학습 (칼만 필터 잡음도 데이터로 맞춤)
    table, mon = diagnose(pipe, recs)         # 로봇 1대 주기별 진단 (표는 '분' 단위)
    metrics = cross_validate(data)            # 로봇(베어링) 단위 교차검증: 한 로봇의 창이 학습과 평가에 나뉘어 들어가지 않음

평가를 로봇 단위로 나누는 이유: 같은 로봇·베어링의 창을 학습과 평가에 섞으면 그 장비만의 특성을 외워
성능이 크게 부풀려진다 (Vieira et al., arXiv:2509.22267, 2025).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import config as C
from pipeline.dataio import DataError, choose_baseline
from pipeline.features import extract
from pipeline.models import NORMAL_LABEL, health, kalman_forecast, label_ko, trend_forecast
from pipeline.system import AMRPipeline


def rows_of(recs, robot):
    rows = [extract(r) for r in recs]
    for r in rows:
        r["robot"] = robot
    return rows


# ---------------------------------------------------------------- 학습
def train(data, units=None, baseline=None, tune=True):
    """units 로봇들로 파이프라인 학습 → (파이프라인, 안내 문구 목록)"""
    names = list(units) if units is not None else list(data.robots)
    baseline = choose_baseline(data, baseline)
    notes, normal, fault, skipped = [], [], [], []
    # 로봇을 정상/고장으로 분류: 고장 라벨이 하나도 없으면 정상 로봇 (라벨이 아예 없는 로봇도 정상으로 취급)
    for name in names:
        rows = rows_of(data.robots[name], name)
        if not any(r["true_sev"] > 0 for r in rows):
            normal += rows
        elif baseline == "per_unit" and any(r["true_sev"] > 0 and r["period"] < C.BASELINE_PERIODS for r in rows):
            skipped.append(name)
        else:
            fault += rows
    if skipped:
        notes.append(f"처음 {C.minutes(C.BASELINE_PERIODS):g}분에 고장 라벨이 있어 학습에서 뺀 로봇: {', '.join(skipped)}")
    if not any(data.robots[n] and any(r["labeled"] for r in data.robots[n]) for n in names):
        notes.append("라벨이 없어 모든 로봇을 정상으로 보고 학습했습니다. 고장 기록이 섞여 있으면 고장을 정상으로 배우게 됩니다.")
    if not normal:                                         # 정상 로봇이 따로 없으면 고장 로봇의 고장 전 구간을 씀
        normal = [dict(r, robot=r["robot"]) for r in fault if r["true_sev"] == 0]
    n_use = sum(1 for r in normal if not r["bump"] and (baseline == "reference" or r["period"] >= C.BASELINE_PERIODS))
    if n_use < C.MIN_TRAIN_WINDOWS:
        raise DataError(f"학습에 쓸 정상 창이 {n_use}개뿐입니다. 정상 구간이 {C.MIN_TRAIN_WINDOWS}초 이상 필요합니다.")

    n_normal_units = len({r["robot"] for r in normal})
    if baseline == "reference" and n_normal_units < 3:   # 공통 기준은 장비마다 다른 정상 상태를 담아야 함
        notes.append(f"공통 정상 기준을 정상 로봇 {n_normal_units}대로만 만들었습니다. 처음 보는 정상 로봇도 이상으로 보일 수 있으니 "
                     "정상 로봇을 3대 이상(가능하면 5대 이상) 넣으세요.")
    pipe = AMRPipeline().fit(normal, fault, baseline)
    b = pipe.bundle
    b["trained_units"] = sorted({r["robot"] for r in normal + fault})
    b["period_s"] = C.PERIOD_S                             # 진단할 때 같은 판정 주기를 쓰도록 모델에 저장
    notes.append(f"판정 주기 {C.PERIOD_S:g}초 · 정상 기준: "
                 f"{f'로봇마다 처음 {C.minutes(C.BASELINE_PERIODS):g}분' if baseline == 'per_unit' else '정상 로봇들의 공통 기준'}"
                 f" · 학습 로봇 {len(b['trained_units'])}대 · 정상 창 {n_use:,}개 · 고장 창 {sum(r['true_sev'] > 0 for r in fault):,}개")
    if b["scale"] == "normal_only":
        notes.append("고장 데이터가 없어 건강 점수는 정상 범위만 기준으로 삼습니다. 정지 위험 판정과 남은 시간 예측은 하지 않습니다.")
    elif b["scale"] == "all_fault":
        notes.append(f"고장 정도 {C.FAIL_CAL_BAND[0]}~{C.FAIL_CAL_BAND[1]} 구간의 창이 없어 고장 창 전체의 중앙값을 9점으로 맞췄습니다.")
    ae = b["detectors"]["ae"]
    if b["scale"] != "normal_only" and ae.rfail <= ae.r995 * C.MIN_FAIL_RATIO:
        notes.append("고장 창의 이상 점수가 정상 범위와 충분히 구분되지 않아 9점 눈금을 맞추지 못했습니다 "
                     "(3점 위도 선형 눈금으로 늘려 씀). 데이터와 라벨을 확인하세요.")
    notes.append("원인 분류 모델 없음 (고장 라벨이 필요)" if b["cause"] is None else
                 f"원인 분류 항목: {', '.join(label_ko(c) for c in b['cause'].classes)}")
    if not b["bat"].active:
        notes.append("배터리: battery 열이 없어 배터리 소모 이상은 판단하지 않습니다.")
    elif b["battery"] is None:
        notes.append("배터리: 정상 소모 측정이 5번 미만이라 경고 기준을 정하지 못했습니다. 더 긴 정상 기록을 넣으세요.")
    else:
        bl = b["battery"]
        notes.append(f"배터리: 정상 측정 {bl['n']}번 기준, 예상보다 {bl['threshold']:.2f}배 넘게 닳는 측정이 "
                     f"{C.BATTERY_ALERT_COUNT}번 연속이면 '배터리 소모 이상' (정상 중앙값 {bl['median']:.2f}배)")
        if bl["n"] < 20:
            notes.append(f"배터리: 정상 측정이 {bl['n']}번뿐이라 기준이 불안정합니다. 정상 운행 기록을 더 길게(잔량 20% 이상 소모) 넣으세요.")
    if tune and b["scale"] != "normal_only":
        notes.append(tune_kalman(pipe, data, sorted({r["robot"] for r in fault})))
    b["notes"] = notes
    return pipe, notes


def _forecast_errors(scores, fn):
    """첫 주의(3점)부터 9점 도달 직전까지 매 주기 예측해 실제 도달 주기와 비교 → [(오차[주기], 90% 범위 안인가)]"""
    hit = next((i for i, s in enumerate(scores) if s >= C.THRESH["danger"]), None)
    start = next((i for i, s in enumerate(scores) if s >= C.THRESH["caution"]), None)
    if hit is None or start is None or start >= hit:
        return []
    out = []
    for i in range(start, hit):
        fc = fn(scores[:i + 1])
        if fc:
            inside = fc["periods_lo"] <= hit - i <= fc["periods_hi"] if "periods_lo" in fc else None
            out.append((abs(i + fc["periods"] - hit), inside))
    return out


def tune_kalman(pipe, data, names):
    """학습 고장 로봇의 주기별 점수로 칼만 필터 잡음(q_level, q_slope, r)을 격자 탐색해 예측 오차가 가장 작은 값을 고른다"""
    series = []
    for n in names:
        try:
            series.append([s for _, s in diagnose(pipe, data.robots[n])[1].period_scores])
        except DataError:
            continue
    best, best_err, n_err = dict(C.KALMAN), None, 0
    for ql, qs, r in itertools.product(*C.KALMAN_GRID.values()):
        cfg = dict(q_level=ql, q_slope=qs, r=r)
        errs = [e for s in series for e, _ in _forecast_errors(s, lambda y: kalman_forecast(y, cfg=cfg))]
        if len(errs) >= 5 and (best_err is None or np.mean(errs) < best_err):
            best, best_err, n_err = cfg, float(np.mean(errs)), len(errs)
    pipe.bundle["kalman"] = best
    if best_err is None:
        return "칼만 필터: 9점까지 간 고장 기록이 부족해 초기값을 그대로 씁니다."
    return f"칼만 필터: 학습 고장 기록 {len(series)}대로 잡음 값을 맞췄습니다 {best} (예측 {n_err}회, 학습 데이터 오차 {C.minutes(best_err):.1f}분)"


# ---------------------------------------------------------------- 진단
def diagnose(pipe, recs, detector="ae"):
    """로봇 1대를 처음부터 끝까지 진단 → (판정 주기별 표, 추론기). 표의 시간·남은 시간은 '분' 단위"""
    if not recs or (pipe.bundle["baseline"] == "per_unit" and recs[-1]["period"] < C.BASELINE_PERIODS):
        raise DataError(f"기록이 짧습니다. 로봇마다 기준을 만드는 방식에서는 {C.minutes(C.BASELINE_PERIODS + 1):g}분 이상이 필요합니다.")
    mon, rows, last, truth = pipe.monitor(detector), [], None, {}

    def flush():
        bat = dict(mon.battery_periods)
        for period, score in mon.period_scores[len(rows):]:
            fc = mon.forecast
            ratio = bat.get(period)
            rows.append({"경과(분)": round(C.minutes(period + 1), 2), "건강 점수": round(score, 2), "상태": C.STATUS_KO[mon.status],
                         "정지 위험까지(분)": round(C.minutes(fc["periods"]), 1) if fc else None,
                         "빠르면(분)": round(C.minutes(fc["periods_lo"]), 1) if fc else None,
                         "추정 원인": (last or {}).get("cause") or "", "판단 근거": ", ".join(n for n, _ in mon.reason),
                         "배터리 소모(예상 대비 배)": round(ratio, 2) if ratio is not None else None,
                         "배터리": "소모 이상" if mon.battery_alert else ("정상" if mon.b["battery"] else "판단 안 함"),
                         "파일 라벨": truth.get(period, "")})

    for rec in recs:
        o = mon.update(rec)
        if rec["labeled"]:
            kind = C.KIND_KO.get(rec["true_kind"], rec["true_kind"])
            truth[rec["period"]] = f"{kind} {rec['true_sev']:.2f}" if rec["true_sev"] > 0 else kind
        flush()
        if o["phase"] == "monitor":
            last = o
    mon.finish()
    flush()
    return pd.DataFrame(rows), mon


# ---------------------------------------------------------------- 평가
def make_folds(data, names):
    """로봇 단위 겹 나누기. 로봇이 CV_MAX_FOLDS대 이하이면 한 대씩, 많으면 라벨별로 고르게 섞어 CV_MAX_FOLDS겹"""
    if len(names) <= C.CV_MAX_FOLDS:
        return [[n] for n in names]
    folds = [[] for _ in range(C.CV_MAX_FOLDS)]
    by_label = {}
    for n in sorted(names):
        by_label.setdefault(str(data.label_of(n)), []).append(n)
    i = 0
    for group in by_label.values():
        for n in group:
            folds[i % C.CV_MAX_FOLDS].append(n)
            i += 1
    return [f for f in folds if f]


def hi_metrics(trajs):
    """건강 지표 품질 (MathWorks Predictive Maintenance 정의): 단조성·추세 일치도·고장 시점 일관성. 0~1, 클수록 좋음"""
    trajs = [np.asarray(t, float) for t in trajs if len(t) >= 3]
    if not trajs:
        return None
    mono = float(np.mean([abs(np.sum(np.diff(t) > 0) - np.sum(np.diff(t) < 0)) / (len(t) - 1) for t in trajs]))
    out = dict(monotonicity=round(mono, 3), units=len(trajs))
    if len(trajs) >= 2:
        grid = np.linspace(0, 1, 20)
        res = [np.interp(grid, np.linspace(0, 1, len(t)), t) for t in trajs]
        cors = [abs(np.corrcoef(a, b)[0, 1]) for a, b in itertools.combinations(res, 2) if np.std(a) > 0 and np.std(b) > 0]
        finals, starts = np.array([t[-1] for t in trajs]), np.array([t[0] for t in trajs])
        out["trendability"] = round(float(min(cors)), 3) if cors else None
        out["prognosability"] = round(float(np.exp(-np.std(finals) / (np.mean(np.abs(finals - starts)) + 1e-12))), 3)
    return out


def cross_validate(data, baseline=None, progress=print):
    """로봇 단위 교차검증 → metrics dict. 각 겹에서 평가 로봇은 학습에 전혀 쓰지 않는다.

    예) 로봇 A·B·C·D·E → 겹1: B~E로 학습, A 평가 / 겹2: A,C~E로 학습, B 평가 / ... 모든 로봇이 한 번씩 '처음 보는 로봇'이 됨
    겹마다 모델을 새로 학습하므로 시간이 걸린다 (칼만 조정은 생략해 시간을 줄임).
    모으는 것: 창 단위(모델별 건강 점수, 원인 예측) + 로봇 단위(첫 주의·경고 주기, 최종 원인, 주기별 점수 궤적)
    라벨 열이 아예 없는 로봇은 학습 때처럼 정상으로 간주해 오경보를 센다 (label_assumed=True로 표시)."""
    names = [n for n in data.robots if data.robots[n]]
    baseline = choose_baseline(data, baseline)
    folds = make_folds(data, names)
    units, win, skipped = [], {k: [] for k in ("y", "labels", "pred")}, []
    det_scores = {}
    for i, test in enumerate(folds, 1):
        try:
            pipe, _ = train(data, [n for n in names if n not in test], baseline, tune=False)
        except DataError as e:
            skipped.append(f"{', '.join(test)}: 학습 불가 ({e})")
            continue
        progress(f"  교차검증 {i}/{len(folds)}: 평가 로봇 {', '.join(test)}")
        b = pipe.bundle
        for name in test:
            recs = data.robots[name]
            try:
                table, mon = diagnose(pipe, recs)
            except DataError as e:
                skipped.append(f"{name}: {e}")
                continue
            # 창 단위: 이 겹의 모델로 평가 로봇의 z를 구해 모델 5종의 건강 점수를 모두 계산 (AUC 비교용)
            rows = rows_of(recs, name)
            pipe.add_residuals(rows)
            zrows, Z = pipe.unit_z(rows)
            label, assumed = data.label_of(name), False
            if label is None:                              # 라벨이 없는 로봇: 학습 때처럼 정상으로 간주
                label, assumed = NORMAL_LABEL, True
            lab = np.array([r["labeled"] or assumed for r in zrows], bool)
            if len(Z) and lab.any():
                y = np.array([r["true_sev"] > 0 for r in zrows])[lab]
                win["y"].append(y)
                for k, d in b["detectors"].items():
                    det_scores.setdefault(k, []).append(health(d, Z[lab]))
                if b["cause"] is not None:
                    keep = np.array([r["true_sev"] == 0 or r["true_sev"] >= C.CAUSE_MIN_SEVERITY for r in zrows]) & lab
                    win["labels"] += [NORMAL_LABEL if r["true_sev"] == 0 else f"{r['true_kind']}|{r['true_side']}"
                                      for r, k in zip(zrows, keep) if k]
                    win["pred"] += list(np.array(b["cause"].classes)[np.argmax(b["cause"].proba(Z[keep]), 1)]) if keep.any() else []
            # 로봇 단위: 주기별 실제 고장 정도와 진단 이벤트를 맞대어 '언제 알아챘나'를 기록
            sev = {}
            for r in recs:
                sev[r["period"]] = max(sev.get(r["period"], 0.0), r["true_sev"])
            first = lambda st: next((period for period, s in mon.events if s >= st), None)
            onset = next((l for l in sorted(sev) if sev[l] > 0), None)
            fail = next((l for l in sorted(sev) if sev[l] >= C.FAILURE_SEVERITY), None)
            scores = [s for _, s in mon.period_scores]
            units.append(dict(robot=name, label=label, label_assumed=assumed,
                              label_ko=label_ko(label) + (" (라벨 없음, 정상으로 간주)" if assumed else ""),
                              onset=onset, caution=first(1), warning=first(2), fail=fail,
                              lead_periods=(fail - first(2)) if fail is not None and first(2) is not None else None,
                              max_score=round(max(scores), 2) if scores else None,
                              cause_pred=label_ko(mon.last_cause_label) if mon.last_cause_label else None,
                              cause_ok=(mon.last_cause_label == label) if (mon.last_cause_label and label and label != NORMAL_LABEL) else None,
                              period_scores=[[int(l), round(s, 3)] for l, s in mon.period_scores],
                              battery_alert=next((l for l, on in mon.battery_events if on), None),
                              battery_max_ratio=round(max((r for _, r in mon.battery_periods if r is not None), default=0.0), 2) or None,
                              sev_range=round(max(sev.values()) - min(v for v in sev.values()), 2) if sev else 0.0,
                              forecast_kalman=_forecast_errors(scores, lambda s: kalman_forecast(s, cfg=b["kalman"])),
                              forecast_trend=_forecast_errors(scores, trend_forecast)))
    return summarize(units, win, det_scores, skipped, baseline, len(folds))


def summarize(units, win, det_scores, skipped, baseline, n_folds):
    """교차검증 결과 → 발표용 수치.
    detectors: 모델별 AUC(고장/정상 창 구분력), 3점 이상 비율(고장 창 감지율·정상 창 오경보율)
    units    : 고장 로봇 중 주의 이상이 뜬 수, 정상 로봇 중 오경보 난 수, 정지 위험(고장 정도 0.8) 전 경고 수와 여유 시간[분]
    cause    : 원인 분류 정확도(창·로봇)와 혼동행렬,  forecast: 칼만 vs 추세선 남은 시간 예측 오차[분]
    battery  : 배터리 소모 이상 경고 — 정상 로봇 오경보 수, battery 라벨 로봇 감지 수, 다른 고장 로봇에서 함께 뜬 수
    health_indicator: 고장 정도가 0.3 이상 변한(진행형) 로봇의 점수 궤적 품질"""
    from pipeline.models import make_detectors
    names = {k: d.name for k, d in make_detectors().items()}
    y = np.concatenate(win["y"]) if win["y"] else np.array([], bool)
    det = {}
    for k, parts in det_scores.items():
        s = np.concatenate(parts)
        both = 0 < y.sum() < len(y)
        det[k] = dict(name=names[k], auc=round(float(roc_auc_score(y, s)), 3) if both else None,
                      recall=round(float(np.mean(s[y] >= C.THRESH["caution"])), 3) if y.any() else None,
                      false_positive=round(float(np.mean(s[~y] >= C.THRESH["caution"])), 3) if (~y).any() else None)
    fault_u = [u for u in units if u["label"] and u["label"] != NORMAL_LABEL]
    normal_u = [u for u in units if u["label"] == NORMAL_LABEL]
    leads = [u["lead_periods"] for u in fault_u if u["lead_periods"] is not None]
    cause_u = [u for u in fault_u if u["cause_ok"] is not None]
    k_err = [e for u in fault_u for e, _ in u["forecast_kalman"]]
    k_in = [i for u in fault_u for _, i in u["forecast_kalman"] if i is not None]
    t_err = [e for u in fault_u for e, _ in u["forecast_trend"]]
    progressive = [[s for _, s in u["period_scores"] if u["onset"] is None or _ >= u["onset"]] for u in fault_u if u["sev_range"] >= 0.3]
    bat_fault = [u for u in fault_u if u["label"].startswith("battery|")]       # 라벨이 battery인 로봇 = 배터리 이상 소모 사례
    m = lambda v: round(C.minutes(float(v)), 1)            # 주기 → 분
    battery = dict(normal_alarm=f"{sum(u['battery_alert'] is not None for u in normal_u)}/{len(normal_u)}",
                   battery_fault_detected=f"{sum(u['battery_alert'] is not None for u in bat_fault)}/{len(bat_fault)}" if bat_fault else None,
                   other_fault_alarm=f"{sum(u['battery_alert'] is not None for u in fault_u if u not in bat_fault)}/{len(fault_u) - len(bat_fault)}")
    cm_labels = sorted(set(win["labels"]) | set(win["pred"]), key=lambda c: (c == NORMAL_LABEL, c))
    return dict(
        protocol=dict(split="로봇(베어링) 단위", folds=n_folds, baseline=baseline, period_s=C.PERIOD_S,
                      note="각 겹에서 평가 로봇의 데이터는 학습·기준·눈금 보정에 전혀 쓰지 않음"),
        windows=dict(n=int(len(y)), fault=int(y.sum())),
        detectors=det,
        units=dict(fault=len(fault_u), normal=len(normal_u), normal_assumed=sum(u["label_assumed"] for u in normal_u),
                   fault_detected=f"{sum(u['caution'] is not None for u in fault_u)}/{len(fault_u)}",
                   false_alarm=f"{sum(u['caution'] is not None for u in normal_u)}/{len(normal_u)}",
                   warned_before_fail=f"{len(leads)}/{sum(u['fail'] is not None for u in fault_u)}",
                   mean_lead_min=m(np.mean(leads)) if leads else None),
        cause=dict(window_accuracy=round(float(np.mean(np.array(win["labels"]) == np.array(win["pred"]))), 3) if win["labels"] else None,
                   unit_accuracy=f"{sum(u['cause_ok'] for u in cause_u)}/{len(cause_u)}" if cause_u else None,
                   labels=cm_labels,
                   confusion=confusion_matrix(win["labels"], win["pred"], labels=cm_labels).tolist() if win["labels"] else None),
        forecast=dict(kalman=dict(mae_min=m(np.mean(k_err)) if k_err else None, n=len(k_err),
                                  coverage=round(float(np.mean(k_in)), 2) if k_in else None),
                      trend=dict(mae_min=m(np.mean(t_err)) if t_err else None, n=len(t_err))),
        health_indicator=hi_metrics(progressive),
        battery=battery,
        unit_results=[{k: v for k, v in u.items() if not k.startswith("forecast")} for u in units],
        skipped=skipped)
