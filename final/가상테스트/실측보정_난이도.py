"""[역할] 실측 보정 가상 로봇으로 한 번 학습한 뒤, 시험 조건을 점점 어렵게 해 성능이 어디서 떨어지는지 본다 (11/18)

100%가 나온 시험(실측보정_시험.py)은 시험 로봇이 61대뿐이고 고장이 충분히 커서 쉬웠다. 실제 현장에 가까운 조건을 단계별로 더한다.
    L0 기본            : 실측보정_시험과 같은 조건 (시험 로봇만 새로 만듦)
    L1 약한 고장        : 시험 로봇의 고장 세기 × 0.6
    L2 L1 + 엔코더 끊김 : 시험 로봇 40%에서 바퀴 하나의 엔코더가 가끔(평균 40초마다 1~8초) 멈춤 — 실측 앞 오른쪽 엔코더에서 본 것
    L3 L2 + 더 약한 고장 : 고장 세기 × 0.4
    L4 L3 + 처음 보는 고장 비율 증가 : 시험 고장의 절반을 학습에 없던 베어링 주파수(120·320 Hz)로
각 단계 시험 로봇 약 80대 (고장 약 50 · 정상 약 30). 채점: 지금 기본 설정과 추천 설정(오토인코더·랜덤포레스트 중 큰 쪽 + 바퀴 점수,
상위 25% 지점 · 1번 · 3점)을 실제 진단 코드(workflow.diagnose)로. 결과: 가상테스트_결과/실측보정_난이도/ (GitHub에 없음)

    python 가상테스트/실측보정_난이도.py --g3 ... --g3 ... --yg ... --fault ... [--jobs 8]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from pipeline import dataio as D, workflow as W  # noqa: E402
from pipeline.models import NORMAL_LABEL  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402
import 실측보정_시험 as T  # noqa: E402
from compare_ae_rf import add_rf  # noqa: E402

LEVELS = [("L0 기본", dict()),
          ("L1 약한 고장 (×0.6)", dict(strength=0.6)),
          ("L2 L1 + 엔코더 끊김", dict(strength=0.6, enc_glitch=0.4)),
          ("L3 L2 + 더 약한 고장 (×0.4)", dict(strength=0.4, enc_glitch=0.4)),
          ("L4 L3 + 처음 보는 베어링 주파수 절반", dict(strength=0.4, enc_glitch=0.4, unseen=0.5))]
CONFIGS = [("지금 기본 (오토인코더 + 바퀴 점수 · 중앙값 · 2번)", ("ae_wheel", 0.5, 2, 3.0)),
           ("추천 (오토인코더·랜덤포레스트 중 큰 쪽 + 바퀴 점수 · 상위 25% · 1번)", ("ae_rf_wheel", 0.75, 1, 3.0)),
           ("오토인코더 + 랜덤포레스트 (바퀴 점수 없음 · 상위 25% · 1번)", ("ae_rf", 0.75, 1, 3.0))]


def enc_glitch(df, rng):
    """바퀴 하나의 엔코더가 가끔 멈춤 (그 동안 누적 펄스가 그대로)"""
    w = str(rng.choice(["fl", "fr", "rl", "rr"]))
    col = f"enc_{w}"
    x = df[col].to_numpy().copy()
    n, fs = len(x), R.FS
    i = int(rng.uniform(5, 40) * fs)
    while i < n:
        L = int(rng.uniform(1, 8) * fs)
        x[i:i + L] = x[i - 1]
        i += L + int(rng.exponential(40) * fs)
    df[col] = x
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g3", action="append", required=True)
    p.add_argument("--yg", action="append", required=True)
    p.add_argument("--fault", required=True)
    p.add_argument("--skip", default="114650,114717,110211")
    p.add_argument("--scale", type=float, default=0.6, help="학습 로봇 규모 (실측보정_시험과 같게)")
    p.add_argument("--test-scale", type=float, default=0.8, help="단계마다 시험 로봇 규모 (0.8 ≈ 80대)")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "실측보정_난이도"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJ), str(HERE)])

    pools = R.RealPools({"공3옥상": a.g3, "욧골": a.yg}, [k for k in a.skip.split(",") if k])
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fp = R.RealPools({"욧골": [a.fault]})
    fm = R.FaultModel(sp["train"][("욧골", 5)], [x for _, x in fp.files[("욧골", 5)]])

    # 학습 (실측보정_시험과 같은 seed·규모 → 같은 학습 로봇)
    train = T.make_specs(a.seed, a.scale, "train", sp["train"])
    data = D.Dataset()
    for s in train:
        recs, fs = D._windows(R.make_table(s, sp["train"], idle, fm), s["name"], data, None)
        data.robots[s["name"]], data.fs[s["name"]] = recs, fs
    pipe, _ = W.train(data, [s["name"] for s in train], "per_unit")
    add_rf(pipe, data, [s["name"] for s in train])
    log(f"학습 끝 (학습 로봇 {len(train)}대)")

    from joblib import Parallel, delayed
    res = {}
    for li, (lname, opt) in enumerate(LEVELS):
        rng = np.random.default_rng(a.seed + 500 + li)
        specs = T.make_specs(a.seed + 1000 + li, a.test_scale, "test", sp["test"])
        for s in specs:
            s["name"] = f"L{li}_{s['name']}"
            if s["fault"]:
                s["strength"] *= opt.get("strength", 1.0)
                if s.get("second"):
                    s["second"]["strength"] *= opt.get("strength", 1.0)
                if rng.random() < opt.get("unseen", 0.0):
                    s.update(kind="bearing", bearing_hz=float(rng.choice([120.0, 320.0])))
        tdata = D.Dataset()
        n_gl = 0
        for s in specs:
            df = R.make_table(s, sp["test"], idle, fm)
            if rng.random() < opt.get("enc_glitch", 0.0):
                enc_glitch(df, rng)
                n_gl += 1
            recs, fs = D._windows(df, s["name"], tdata, None)
            tdata.robots[s["name"]], tdata.fs[s["name"]] = recs, fs
        nF = sum(s["fault"] for s in specs)
        res[lname] = dict(robots=len(specs), fault=nF, normal=len(specs) - nF, enc_glitch_robots=n_gl)
        for cname, (k, q, cf, cu) in CONFIGS:
            pipe.bundle["period_quantile"], pipe.bundle["confirm_periods"] = q, cf
            got = Parallel(n_jobs=a.jobs, backend="loky")(delayed(T._diag_one)(pipe, tdata.robots[s["name"]], k, cu) for s in specs)
            hit = fa = kind_ok = kind_n = 0
            for s, (first, lbl) in zip(specs, got):
                if s["fault"]:
                    hit += first is not None and first >= int(s["onset"] * 60 // C.PERIOD_S)
                    kinds = {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())
                    if lbl and lbl != NORMAL_LABEL:
                        kind_n += 1
                        kind_ok += lbl.split("|")[0] in kinds
                else:
                    fa += first is not None
            nN = len(specs) - nF
            res[lname][cname] = dict(detect=round(100 * hit / nF, 1), false_alarm=round(100 * fa / nN, 1),
                                     accuracy=round(100 * (hit + nN - fa) / len(specs), 1),
                                     kind_ok=round(100 * kind_ok / kind_n, 1) if kind_n else None,
                                     hits=f"{hit}/{nF}", false=f"{fa}/{nN}")
            log(f"{lname} · {cname}: {res[lname][cname]}")
        (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "=" * 70)
    for lname, v in res.items():
        print(f"[{lname}] 시험 {v['robots']}대 (고장 {v['fault']} · 정상 {v['normal']} · 엔코더 끊김 {v['enc_glitch_robots']}대)")
        for cname, _ in CONFIGS:
            x = v[cname]
            print(f"   {cname}: 정답률 {x['accuracy']}% · 감지 {x['detect']}% ({x['hits']}) · 오경보 {x['false_alarm']}% ({x['false']}) · 원인 {x['kind_ok']}%")
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
