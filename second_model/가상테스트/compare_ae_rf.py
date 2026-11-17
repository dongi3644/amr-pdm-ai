"""[역할] 가상 로봇으로 '오토인코더 + 랜덤포레스트'(지금 구조)와 '랜덤포레스트만'을 같은 조건에서 비교한다 (코드 점검용, 성능 아님)

    python 가상테스트/compare_ae_rf.py                # 학습 72대 · 시험 51대 (--scale 0.3, 약 25분)
    python 가상테스트/compare_ae_rf.py --scale 0.1    # 빠른 확인

비교하는 4가지 (원인 분류는 넷 다 같은 랜덤포레스트, 다른 것은 '이상을 알아채는 점수'뿐)
    ① 오토인코더 + 바퀴 점수 + 랜덤포레스트   지금 구조 (엔코더가 있을 때 주 모델)
    ② 오토인코더 + 랜덤포레스트               바퀴 점수(규칙)를 뺀 것 → 순수하게 AI 둘만
    ③ 랜덤포레스트만                         랜덤포레스트가 낸 '정상이 아닐 확률' = 이상 점수
    ④ 랜덤포레스트만 + 바퀴 점수              ③에 바퀴 점수를 더한 것
③의 랜덤포레스트는 지금 원인 분류와 같은 설정(나무 150그루, class_weight=balanced)으로, 학습용 정상 로봇 + 고장 로봇의 창으로 배우고,
오토인코더와 똑같이 '학습에 안 쓴 정상 로봇 99.5% = 3점, 고장 정도 0.75~0.85 = 9점' 눈금을 맞춘다. 판정(1분·2번 연속)도 같다.
시험 로봇은 test_complex와 같다 (학습에 없던 고장: 베어링 변형·속도 극단 포함). 가상 로봇에는 엔코더 4개가 항상 있다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from pipeline import dataio as D, workflow as W  # noqa: E402
from pipeline import models as M  # noqa: E402
from pipeline.models import NORMAL_LABEL, CombinedDetector, calibrate, health  # noqa: E402
from pipeline.system import per_unit_z  # noqa: E402
import test_complex as T  # noqa: E402


class RFDetector:
    """랜덤포레스트가 낸 '정상이 아닐 확률'(1 - P(정상))을 이상 점수로 쓰는 비교용 감지기"""
    name = "랜덤포레스트만"

    def fit(self, Z, y):
        self.m = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                        random_state=C.SEED, n_jobs=-1).fit(Z, y)
        self.m.n_jobs = 1
        self.i_normal = list(self.m.classes_).index(NORMAL_LABEL)
        return self

    def raw(self, Z):
        return 1.0 - self.m.predict_proba(Z)[:, self.i_normal]


def add_rf(pipe, data, units):
    """학습 로봇으로 RF 감지기를 만들어 pipe.bundle['detectors']에 'rf', 'rf_wheel'로 넣는다 (system.fit과 같은 로봇 나누기)"""
    b = pipe.bundle
    feats = b["feats"]
    normal, fault = [], []
    for n in units:
        rows = W.rows_of(data.robots[n], n)
        (fault if any(r["true_sev"] > 0 for r in rows) else normal).extend(rows)
    _, Zfit = per_unit_z([r for r in normal if r["robot"] in b["fit_units"]], feats)
    _, Zcal = per_unit_z([r for r in normal if r["robot"] in b["cal_units"]], feats)
    frows, Zf = per_unit_z(fault, feats)
    sev = np.array([r["true_sev"] for r in frows])
    keep = (sev == 0) | (sev >= C.CAUSE_MIN_SEVERITY)
    labels = np.array([NORMAL_LABEL if r["true_sev"] == 0 else f"{r['true_kind']}|{r['true_side']}" for r in frows])
    Z = np.vstack([Zfit, Zf[keep]])
    y = np.concatenate([[NORMAL_LABEL] * len(Zfit), labels[keep]])
    Zfail = Zf[(sev >= C.FAIL_CAL_BAND[0]) & (sev <= C.FAIL_CAL_BAND[1])]
    rf = calibrate(RFDetector().fit(Z, y), Zcal, Zfail)
    b["detectors"]["rf"] = rf
    if "wheel" in b["detectors"]:
        b["detectors"]["rf_wheel"] = calibrate(CombinedDetector(rf, b["detectors"]["wheel"]), Zcal, Zfail)
    return len(Z)


SETUPS = [("ae_wheel", "① 오토인코더 + 바퀴 점수 + 랜덤포레스트 (지금)"), ("ae", "② 오토인코더 + 랜덤포레스트"),
          ("rf", "③ 랜덤포레스트만"), ("rf_wheel", "④ 랜덤포레스트 + 바퀴 점수")]
UNSEEN = {"베어링 변형", "속도 다름"}                     # 시험에서 학습 때 없던 값으로 만든 경우


def main():
    p = argparse.ArgumentParser(description="오토인코더+랜덤포레스트 vs 랜덤포레스트만 (가상, 코드 점검용)")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--scale", type=float, default=0.3)
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "AE_RF_비교"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)

    train_specs, test_specs = T.make_specs(a.seed, a.scale)
    for s in train_specs + test_specs:
        s.update(encoders=True, glitch=0, no_gyro=False)
    data = D.Dataset()
    for s in train_specs + test_specs:
        T.add_robot(data, s)
    log(f"가상 로봇 학습 {len(train_specs)}대 · 시험 {len(test_specs)}대 만듦")
    train_names = [s["name"] for s in train_specs]

    pipe, _ = W.train(data, train_names, "per_unit")
    n_rf = add_rf(pipe, data, train_names)
    log(f"학습 끝 (오토인코더 + 원인 랜덤포레스트 + 비교용 랜덤포레스트 감지기, 창 {n_rf:,}개)")

    rows = {k: [] for k, _ in SETUPS}
    win_y, win_s = [], {k: [] for k, _ in SETUPS}
    for s in test_specs:
        recs = data.robots[s["name"]]
        zrows, Z = pipe.unit_z(W.rows_of(recs, s["name"]))
        if len(Z):
            win_y.append(np.array([r["true_sev"] > 0 for r in zrows]))
            for k, _ in SETUPS:
                win_s[k].append(health(pipe.bundle["detectors"][k], Z))
        kinds = {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())
        for k, _ in SETUPS:
            table, mon = W.diagnose(pipe, recs, detector=k)
            first = next((pp for pp, st in mon.events if st >= 1), None)
            lbl = mon.last_cause_label
            rows[k].append(dict(cat=s["cat"], group=s["group"], fault=bool(s["fault"]), alarm=first is not None,
                                delay=(C.minutes(first + 1) - s["onset"]) if (first is not None and s["fault"]) else None,
                                kind_ok=(lbl.split("|")[0] in kinds) if (s["fault"] and lbl and lbl != NORMAL_LABEL) else None))
        log(f"시험: {s['name']}")

    y = np.concatenate(win_y)
    res = {}
    pc = lambda a_, b_: round(100 * a_ / b_, 1) if b_ else None
    for k, name in SETUPS:
        r = rows[k]
        F = [x for x in r if x["fault"]]
        N = [x for x in r if not x["fault"]]
        U = [x for x in F if x["cat"] in UNSEEN]
        cz = [x for x in F if x["kind_ok"] is not None]
        sc = np.concatenate(win_s[k])
        res[k] = dict(name=name, robots=len(r),
                      accuracy=pc(sum(x["alarm"] for x in F) + sum(not x["alarm"] for x in N), len(r)),
                      detect=pc(sum(x["alarm"] for x in F), len(F)), false_alarm=pc(sum(x["alarm"] for x in N), len(N)),
                      unseen_detect=pc(sum(x["alarm"] for x in U), len(U)), unseen_n=len(U),
                      delay=round(float(np.mean([x["delay"] for x in F if x["delay"] is not None])), 1) if any(x["alarm"] for x in F) else None,
                      kind_ok=pc(sum(x["kind_ok"] for x in cz), len(cz)), cause_n=len(cz),
                      auc=round(float(roc_auc_score(y, sc)), 3),
                      win_detect=round(100 * float(np.mean(sc[y] >= C.THRESH["caution"])), 1),
                      win_false=round(100 * float(np.mean(sc[~y] >= C.THRESH["caution"])), 1),
                      groups={g: dict(acc=pc(sum(x["alarm"] == x["fault"] for x in r if x["group"] == g),
                                             sum(1 for x in r if x["group"] == g))) for g in ["일반", "복잡한 고장", "어려운 정상"]})
    (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    f = lambda v, u="%": "-" if v is None else f"{v:g}{u}"
    nF = sum(x["fault"] for x in rows["ae"])
    lines = ["# 오토인코더 + 랜덤포레스트 vs 랜덤포레스트만 (가상 로봇, 코드 점검용 · 성능 아님)", "",
             f"- 실행: `가상테스트/compare_ae_rf.py --seed {a.seed} --scale {a.scale:g}` · 학습 {len(train_specs)}대 · 시험 {len(test_specs)}대 (고장 {nF} · 정상 {len(test_specs) - nF})",
             "- 원인 분류는 모두 같은 랜덤포레스트, 다른 것은 이상을 알아채는 점수뿐. 가상 로봇에는 엔코더 4개가 항상 있음", "",
             "| 방법 | 판정 정답률 | 고장 감지율 | 오경보율 | 처음 보는 고장 감지 | 고장 → 경미까지 | 고장 종류 정확도 | AUC (1초) | 1초 감지율 / 오경보율 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, _ in SETUPS:
        v = res[k]
        lines.append(f"| {v['name']} | **{f(v['accuracy'])}** | {f(v['detect'])} | {f(v['false_alarm'])} | {f(v['unseen_detect'])} ({v['unseen_n']}대) | "
                     f"{f(v['delay'], '분')} | {f(v['kind_ok'])} ({v['cause_n']}대) | {v['auc']} | {f(v['win_detect'])} / {f(v['win_false'])} |")
    lines += ["", "| 방법 | 일반 | 복잡한 고장 | 어려운 정상 |", "|---|---|---|---|"]
    lines += [f"| {res[k]['name']} | " + " | ".join(f(res[k]['groups'][g]['acc']) for g in ["일반", "복잡한 고장", "어려운 정상"]) + " |" for k, _ in SETUPS]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
