"""[역할] 실측혼합_시험.py로 학습한 모델(각속도 없음, 엔코더 4개)에 맞는 판정 설정을 검증 로봇으로 고르고, 시험 로봇으로 확인 (11/19)

    python 가상테스트/실측혼합_조정.py --g3 ... --g3 ... --yg ... --fault ... [--jobs 8]

  1) 저장된 모델(가상테스트_결과/실측혼합/model_bundle.joblib)을 불러옴 (다시 학습하지 않음)
  2) 검증 로봇: 실측 '검증' 몫의 조각으로 L0·L1·L2 조건을 섞어 만듦 (시험 로봇과 겹치지 않음)
  3) '이상 점수 방법 × 1분 점수 지점 × 연속 횟수 × 경미 기준' 조합을 검증 로봇으로 채점 → 오경보 3% 이하에서 정답률 최고 조합
  4) 시험 로봇(L0~L4, 실측 '시험' 몫)을 지금 기본 설정과 고른 설정으로 실제 진단 코드(workflow.diagnose)로 채점
로봇 만들기·진단은 여러 프로세스로 나눠 돌림. 결과: 가상테스트_결과/실측혼합_조정/ (GitHub에 없음)
"""
import argparse
import itertools
import json
import os
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
from pipeline import dataio as D, workflow as W  # noqa: E402
from pipeline.models import NORMAL_LABEL, health  # noqa: E402
from pipeline.system import AMRPipeline  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402
import 실측보정_시험 as T  # noqa: E402
from 실측보정_난이도 import LEVELS, enc_glitch  # noqa: E402
from 실측혼합_시험 import GYRO, TRAIN_FAULT  # noqa: E402

GRID = dict(quantile=[0.5, 0.75, 0.9], confirm=[1, 2], caution=[1.5, 2.0, 2.5, 3.0])
DEFAULT = (None, C.PERIOD_QUANTILE, C.CONFIRM_PERIODS, C.THRESH["caution"])


def _make(s, pools, idle, fm, opt, seed):
    """로봇 1대 만들기 (프로세스 하나에서) → (1초 창, 샘플링)"""
    rng = np.random.default_rng(seed)
    df = R.make_table(s, pools, idle, fm).drop(columns=GYRO)
    if rng.random() < opt.get("enc_glitch", 0.0):
        enc_glitch(df, rng)
    data = D.Dataset()
    return D._windows(df, s["name"], data, None)


def build(specs, opt, pools, idle, fm, seed, jobs, rng):
    from joblib import Parallel, delayed
    for s in specs:
        if s["fault"]:
            s["strength"] *= opt.get("strength", 1.0)
            if s.get("second"):
                s["second"]["strength"] *= opt.get("strength", 1.0)
            if rng.random() < opt.get("unseen", 0.0):
                s.update(kind="bearing", bearing_hz=float(rng.choice([120.0, 320.0])))
    got = Parallel(n_jobs=jobs, backend="loky")(delayed(_make)(s, pools, idle, fm, opt, seed + i) for i, s in enumerate(specs))
    return {s["name"]: g[0] for s, g in zip(specs, got)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g3", action="append", required=True)
    p.add_argument("--yg", action="append", required=True)
    p.add_argument("--fault", required=True)
    p.add_argument("--skip", default="114650,114717,110211")
    p.add_argument("--model", default=str(PROJ / "가상테스트_결과" / "실측혼합" / "model_bundle.joblib"))
    p.add_argument("--val-scale", type=float, default=0.6)
    p.add_argument("--test-scale", type=float, default=0.8)
    p.add_argument("--max-fa", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "실측혼합_조정"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJ), str(HERE)])

    pools = R.RealPools({"공3옥상": a.g3, "욧골": a.yg}, [k for k in a.skip.split(",") if k])
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    tf = [pd.read_csv(Path(a.fault) / f"{k}.csv") for k in TRAIN_FAULT]
    fm = R.FaultModel(sp["train"][("욧골", 5)], [np.column_stack([d.ax / R.LSB, d.ay / R.LSB, d.az / R.LSB, d.mA / 1000, d.mV / 1000]) for d in tf])
    pipe = AMRPipeline.load(a.model)
    b = pipe.bundle
    keys = [k for k in ("ae_wheel", "ae_rf_wheel", "ae_rf", "rf", "ae") if k in b["detectors"]]
    log(f"모델 불러옴: 주 모델 {b['main']} · 특징 {len(b['feats'])}개 · 비교할 방법 {keys}")

    def cache_of(robots):
        c = {}
        for name, recs in robots.items():
            zrows, Z = pipe.unit_z(W.rows_of(recs, name))
            c[name] = {"period": np.array([z["period"] for z in zrows]), **{k: health(b["detectors"][k], Z) for k in keys}}
        return c

    # 2) 검증 로봇 (L0·L1·L2 섞음)
    val_specs, val_cache = [], {}
    rng = np.random.default_rng(a.seed + 3000)
    for li in (0, 1, 2):
        specs = T.make_specs(a.seed + 2000 + li, a.val_scale, "val", sp["val"])
        for s in specs:
            s["name"] = f"검증L{li}_{s['name']}"
        robots = build(specs, LEVELS[li][1], sp["val"], idle, fm, a.seed + 4000 + 100 * li, a.jobs, rng)
        val_cache.update(cache_of(robots))
        val_specs += specs
        log(f"검증 로봇 {LEVELS[li][0]} {len(specs)}대")
    grid = []
    for k, q, cf, cu in itertools.product(keys, GRID["quantile"], GRID["confirm"], GRID["caution"]):
        sm = T.summary(T.score_robots(val_cache, val_specs, k, q, cf, cu))
        grid.append(dict(setup=k, quantile=q, confirm=cf, caution=cu, **sm))
    ok = [g for g in grid if g["false_alarm"] <= a.max_fa] or grid
    best = max(ok, key=lambda g: (g["accuracy"], -g["false_alarm"], -(g["delay"] or 99)))
    grid.sort(key=lambda g: (-g["accuracy"], g["false_alarm"]))
    log(f"검증 최고 (오경보 {a.max_fa}% 이하): {best}")
    dflt = next(g for g in grid if g["setup"] == b["main"] and g["quantile"] == DEFAULT[1] and g["confirm"] == DEFAULT[2] and g["caution"] == DEFAULT[3])
    log(f"검증에서 지금 기본 설정: {dflt}")

    # 4) 시험 로봇 (L0~L4), 실제 진단 코드로
    from joblib import Parallel, delayed
    pick = (best["setup"], best["quantile"], best["confirm"], best["caution"])
    configs = [("지금 기본 설정", (b["main"],) + DEFAULT[1:]), ("검증으로 고른 설정", pick)]
    res = dict(best_val=best, default_val=dflt, top_val=grid[:15], levels={})
    rng = np.random.default_rng(a.seed + 5000)
    for li, (lname, opt) in enumerate(LEVELS):
        specs = T.make_specs(a.seed + 1000 + li, a.test_scale, "test", sp["test"])
        for s in specs:
            s["name"] = f"시험L{li}_{s['name']}"
        robots = build(specs, opt, sp["test"], idle, fm, a.seed + 6000 + 100 * li, a.jobs, rng)
        res["levels"][lname] = {}
        for cname, (k, q, cf, cu) in configs:
            pipe.bundle["period_quantile"], pipe.bundle["confirm_periods"] = q, cf
            got = Parallel(n_jobs=a.jobs, backend="loky")(delayed(T._diag_one)(pipe, robots[s["name"]], k, cu) for s in specs)
            hit = fa = ko = kn = 0
            nF = sum(s["fault"] for s in specs)
            for s, (first, lbl) in zip(specs, got):
                if s["fault"]:
                    hit += first is not None and first >= int(s["onset"] * 60 // C.PERIOD_S)
                    kinds = {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())
                    if lbl and lbl != NORMAL_LABEL:
                        kn += 1
                        ko += lbl.split("|")[0] in kinds
                else:
                    fa += first is not None
            nN = len(specs) - nF
            v = dict(setup=k, quantile=q, confirm=cf, caution=cu, robots=len(specs),
                     accuracy=round(100 * (hit + nN - fa) / len(specs), 1), detect=round(100 * hit / nF, 1),
                     false_alarm=round(100 * fa / nN, 1), kind_ok=round(100 * ko / kn, 1) if kn else None,
                     hits=f"{hit}/{nF}", false=f"{fa}/{nN}", causes=f"{ko}/{kn}")
            res["levels"][lname][cname] = v
            log(f"{lname} · {cname}: {v}")
        (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"검증으로 고른 설정: {pick} (검증 정답률 {best['accuracy']}% · 오경보 {best['false_alarm']}%)")
    for lname, v in res["levels"].items():
        print(f"[{lname}]")
        for cname, x in v.items():
            print(f"   {cname}: 정답률 {x['accuracy']}% · 감지 {x['detect']}% ({x['hits']}) · 오경보 {x['false_alarm']}% ({x['false']}) · 원인 {x['kind_ok']}% ({x['causes']})")
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
