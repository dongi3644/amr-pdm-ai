"""[역할] third_model(1D-CNN) 성능 시험: 실측 보정 가상 로봇으로 학습 → 아는 고장 · 처음 보는 고장을 따로 채점 (11/19)

    python 가상테스트/성능시험.py --g3 "data/정상 데이터_공3옥상" --g3 "data/정상 데이터_공3옥상(11.16)" \
        --yg "data/정상 데이터_욧골(11.15)" --fault "data/고장 데이터_욧골" [--jobs 8]

가상 로봇: second_model/가상테스트/실측보정_로봇.py (실측 정상 조각을 이어 붙이고, 실측 테이프 고장의 진동 증가분을 넣음).
  엔코더 4개(가정), 각속도 없음. 실측 파일은 학습·검증·시험으로 나눠 겹치지 않음.
학습: 정상 + 아는 고장 3종(바퀴 불균형·체결 풀림·마찰 증가) × 바퀴 4개 + 가상 이상(config.SYNTH_ANOMALY)
시험 (엔코더 끊김 없음)
  K  아는 고장만 (학습과 같은 3종, 처음 보는 로봇)
  U  고장 로봇의 절반을 '처음 보는 고장'으로 바꿈: 기어 마모(gear) · 바퀴 찍힘(flat) · 학습에 없던 주파수의 베어링(120·160 Hz)
판정 설정(1분 점수 지점 · 연속 횟수 · 경미 경보 점수)은 검증 로봇(아는 고장만, 시험과 겹치지 않음)으로 고름.
가상 로봇 결과라 실측 성능이 아님. 결과: 가상테스트_결과/성능시험/ (GitHub에 없음)
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
GEN = PROJ.parent / "second_model" / "가상테스트"        # 실측 보정 가상 로봇 생성기 (numpy·pandas만 씀)
sys.path.insert(0, str(PROJ))
sys.path.append(str(GEN))
import config as C  # noqa: E402
from pipeline import dataio as D, workflow as W  # noqa: E402
from pipeline.models import NORMAL_LABEL, health  # noqa: E402
from pipeline.system import AMRPipeline  # noqa: E402
import sim_robots as S  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402

GYRO = ["gx", "gy", "gz"]
TRAIN = {"정상": 18, "바닥 바뀜": 5, "짐 실음": 4, "저전압 정상": 3, "속도 바뀜": 3,
         "급성 고장": 6, "간헐 고장": 6, "복합 고장": 8, "약한 고장": 6, "속도 다름": 4}
EVAL_COMPLEX = {"급성 고장": 5, "간헐 고장": 5, "복합 고장": 5, "약한 고장": 5, "고장+저전압": 5}
EVAL_HARD = {"바닥 바뀜": 6, "짐 실음": 4, "저전압 정상": 4, "속도 바뀜": 4}
UNSEEN = ["gear", "flat", "bearing_new"]
UNSEEN_KO = {"gear": "기어 마모", "flat": "바퀴 찍힘", "bearing_new": "처음 보는 주파수의 베어링"}
GRID = dict(quantile=[0.5, 0.75], confirm=[1, 2, 3], caution=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
KEYS = ["ae_wheel", "cnn_ae", "deep", "rf", "deep_rf", "feat_rf", "spec_ae", "spec_wheel", "spec_rf"]


def place(spec, rng, pools, cat):
    yg = ("욧골", 5) in pools and len(pools[("욧골", 5)]) > 0
    if cat == "바닥 바뀜":
        spec.update(floor="공3옥상", spd=5, floor2="욧골" if yg else "공3옥상", floor_at=rng.uniform(7, 10))
    elif cat == "속도 바뀜":
        spec.update(floor="공3옥상", spd=5, spd2=7, spd_at=rng.uniform(7, 10))
    elif yg and rng.random() < 0.35:
        spec.update(floor="욧골", spd=5)
    else:
        spec.update(floor="공3옥상", spd=int(rng.choice([5, 7])) if cat != "속도 다름" else 7)
    if cat in ("저전압 정상", "고장+저전압"):
        spec["v_drop"] = rng.uniform(0.7, 1.1)


def make_specs(seed, scale, role, pools):
    rng = np.random.default_rng(seed)
    nn = lambda v: max(1, int(round(v * scale)))
    specs = []
    if role == "train":
        for k in S.KINDS:
            for w in S.WHEELS:
                specs += [S.complex_spec(rng, f"학습_점진_{k}_{w}_{j + 1}", "점진 고장", k, w) for j in range(nn(2))]
        for cat, cnt in TRAIN.items():
            specs += [S.complex_spec(rng, f"학습_{cat.replace(' ', '')}_{j + 1:02d}", cat if cat != "속도 바뀜" else "정상") | {"cat": cat}
                      for j in range(nn(cnt))]
    else:
        tag = "검증" if role == "val" else "시험"
        for i in range(nn(40)):
            s = S.random_spec(rng, f"{tag}_일반{i + 1:03d}", rng.random() < 0.5)
            specs.append(s | {"cat": "일반 고장" if s["fault"] else "일반 정상"})
        for cat, cnt in EVAL_COMPLEX.items():
            specs += [S.complex_spec(rng, f"{tag}_{cat.replace(' ', '')}_{j + 1:02d}", cat, extreme=True) for j in range(nn(cnt))]
        for cat, cnt in EVAL_HARD.items():
            specs += [S.complex_spec(rng, f"{tag}_{cat.replace(' ', '')}_{j + 1:02d}", cat if cat != "속도 바뀜" else "정상", extreme=True)
                      | {"cat": cat} for j in range(nn(cnt))]
    for s in specs:
        place(s, rng, pools, s["cat"])
    return specs


def make_unseen(specs, rng, share):
    """고장 로봇의 share만큼을 처음 보는 고장으로 (두 번째 고장은 지움: 아는 고장으로 잡히지 않게)"""
    for s in specs:
        s["unseen"] = None
        if s["fault"] and rng.random() < share:
            u = UNSEEN[rng.integers(len(UNSEEN))]
            s.pop("second", None)
            s.update(kind="bearing", bearing_hz=float(rng.choice([120.0, 160.0]))) if u == "bearing_new" else s.update(kind=u)
            s["unseen"] = u
    return specs


def _gen(s, pools, idle, fm):
    df = R.make_table(s, pools, idle, fm).drop(columns=GYRO)
    return D._windows(df, s["name"], D.Dataset(), None)[0]


_PIPE = {}


def _score(path, s, pools, idle, fm):
    """로봇 1대 만들기 → 감지 점수들 + 원인 확률 (프로세스 하나에서, 모델은 파일에서 한 번만 읽음)"""
    try:                                                   # 1D-CNN을 켰을 때만 PyTorch가 필요
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    if path not in _PIPE:
        _PIPE[path] = AMRPipeline.load(path)
    pipe = _PIPE[path]
    rows = W.rows_of(_gen(s, pools, idle, fm), s["name"])
    zrows, Z = pipe.unit_z(rows)
    b = pipe.bundle
    out = {"period": np.array([z["period"] for z in zrows])}
    for k in KEYS:
        if k in b["detectors"]:
            out[k] = health(b["detectors"][k], Z).astype(np.float32)
    if b["cause"] is not None:
        out["cause_p"] = b["cause"].proba(Z).astype(np.float32)
    return out


def fast_status(periods, scores, q, confirm, caution):
    out, state, cand, run = [], 0, 0, 0
    for p in sorted(set(periods)):
        if p < C.BASELINE_PERIODS:
            continue
        v = float(np.quantile(scores[periods == p], q))
        lvl = 2 if v >= C.THRESH["warning"] else (1 if v >= caution else 0)
        cand, run = (cand, run + 1) if lvl == cand else (lvl, 1)
        if run >= confirm and lvl != state:
            state = lvl
        out.append((p, state))
    return out


def cause_of(c, first, classes, window=180):
    """경보가 난 뒤 창들의 원인 확률 평균 → 종류마다 바퀴 확률을 더해 1위 종류"""
    m = c["period"] >= first
    if not m.any():
        return None
    p = c["cause_p"][m][-window:].mean(0)
    sums = {}
    for cl, v in zip(classes, p):
        if cl != NORMAL_LABEL:
            sums[cl.split("|")[0]] = sums.get(cl.split("|")[0], 0.0) + float(v)
    return max(sums, key=sums.get) if sums else None


def evaluate(cache, specs, key, q, cf, cu, classes=None):
    res = dict(known_hit=0, known_n=0, unseen_hit=0, unseen_n=0, fa=0, normal_n=0, cause_ok=0, cause_n=0,
               by_unseen={u: [0, 0] for u in UNSEEN})
    for s in specs:
        c = cache[s["name"]]
        st = fast_status(c["period"], c[key], q, cf, cu)
        first = next((p for p, x in st if x >= 1), None)
        if not s["fault"]:
            res["normal_n"] += 1
            res["fa"] += first is not None
            continue
        hit = first is not None and first >= int(s["onset"] * 60 // C.PERIOD_S)
        u = s.get("unseen")
        if u:
            res["unseen_n"] += 1
            res["unseen_hit"] += hit
            res["by_unseen"][u][0] += hit
            res["by_unseen"][u][1] += 1
        else:
            res["known_n"] += 1
            res["known_hit"] += hit
        if hit and classes is not None and "cause_p" in c:
            k = cause_of(c, first, classes)
            want = {"bearing"} if u == "bearing_new" else {"unknown"} if u else {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())
            if u == "bearing_new":
                want.add("unknown")                         # 처음 보는 주파수의 베어링: '마찰 증가' 또는 '원인 미확정'이면 맞음
            res["cause_n"] += 1
            res["cause_ok"] += k in want
    pc = lambda a, b: round(100 * float(a) / b, 1) if b else None
    nF = res["known_n"] + res["unseen_n"]
    hits = res["known_hit"] + res["unseen_hit"]
    return dict(accuracy=pc(hits + res["normal_n"] - res["fa"], nF + res["normal_n"]), detect=pc(hits, nF),
                known=pc(res["known_hit"], res["known_n"]), unseen=pc(res["unseen_hit"], res["unseen_n"]),
                false_alarm=pc(res["fa"], res["normal_n"]), cause=pc(res["cause_ok"], res["cause_n"]),
                n=f"고장 {nF}(처음 보는 {res['unseen_n']}) · 정상 {res['normal_n']}",
                by_unseen={UNSEEN_KO[u]: f"{a}/{b}" for u, (a, b) in res["by_unseen"].items() if b})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g3", action="append", required=True)
    p.add_argument("--yg", action="append", required=True)
    p.add_argument("--fault", required=True)
    p.add_argument("--skip", default="114650,114717,110211")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--val-scale", type=float, default=0.4)
    p.add_argument("--test-scale", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-fa", type=float, default=3.0)
    p.add_argument("--epochs", type=int, default=15, help="1D-CNN 최대 학습 반복 (config.DEEP_EPOCHS 대신)")
    p.add_argument("--max-windows", type=int, default=20000, help="1D-CNN 학습 창 수 상한 (config.DEEP_MAX_WINDOWS 대신)")
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "성능시험"))
    a = p.parse_args()
    C.DEEP_EPOCHS, C.DEEP_MAX_WINDOWS, C.DEEP_PATIENCE = a.epochs, a.max_windows, min(C.DEEP_PATIENCE, 4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJ), str(GEN), str(HERE)])
    from joblib import Parallel, delayed

    pools = R.RealPools({"공3옥상": a.g3, "욧골": a.yg}, [k for k in a.skip.split(",") if k])
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fp = R.RealPools({"욧골": [a.fault]})
    fm = R.FaultModel(sp["train"][("욧골", 5)], [x for _, x in fp.files[("욧골", 5)]])
    log(f"실측 파일 나누기: " + " / ".join(f"{r} {len(v)}개" for r, v in pools.names.items()))

    train = make_specs(a.seed, a.scale, "train", sp["train"])
    got = Parallel(n_jobs=a.jobs, backend="loky")(delayed(_gen)(s, sp["train"], idle, fm) for s in train)
    data = D.Dataset()
    for s, recs in zip(train, got):
        data.robots[s["name"]], data.fs[s["name"]] = recs, R.FS
    del got
    log(f"학습 로봇 {len(train)}대 만듦 (고장 {sum(s['fault'] for s in train)}대)")
    pipe, notes = W.train(data, [s["name"] for s in train], "per_unit")
    path = out / "model_bundle.joblib"
    pipe.save(path)
    b = pipe.bundle
    log(f"학습 끝: 주 모델 {b['main']} · 특징 {len(b['feats'])}개 · 파형 {b.get('wave_ch')} · 감지 {list(b['detectors'])}")
    for n_ in notes:
        log("  " + n_)
    del data
    classes = b["cause"].classes if b["cause"] is not None else None
    keys = [k for k in KEYS if k in b["detectors"]]

    def cache_of(specs, pools):
        res = Parallel(n_jobs=a.jobs, backend="loky")(delayed(_score)(str(path), s, pools, idle, fm) for s in specs)
        return {s["name"]: r for s, r in zip(specs, res)}

    val = make_specs(a.seed + 1, a.val_scale, "val", sp["val"])
    for s in val:
        s["unseen"] = None
    vc = cache_of(val, sp["val"])
    log(f"검증 로봇 {len(val)}대 점수 계산 끝 (아는 고장만)")
    pick = {}
    for k in keys:
        grid = [dict(q=q, cf=cf, cu=cu, **evaluate(vc, val, k, q, cf, cu)) for q, cf, cu in itertools.product(*GRID.values())]
        ok = [g for g in grid if g["false_alarm"] <= a.max_fa] or grid
        pick[k] = max(ok, key=lambda g: (g["accuracy"], -g["false_alarm"], -g["cu"], -g["cf"]))   # 같으면 더 민감한(낮은) 문턱
        log(f"  검증 최고 {k}: 1분 점수 {pick[k]['q']} · {pick[k]['cf']}번 · 경미 {pick[k]['cu']}점 → 정답률 {pick[k]['accuracy']}% · 오경보 {pick[k]['false_alarm']}%")

    res = dict(main=b["main"], feats=len(b["feats"]), wave_ch=b.get("wave_ch"), notes=notes, pick=pick, levels={})
    rng = np.random.default_rng(a.seed + 500)
    for li, (lname, share, strength) in enumerate([("K 아는 고장만", 0.0, 1.0), ("U 고장 절반이 처음 보는 고장", 0.5, 1.0),
                                                     ("U2 약한 고장(×0.6) + 절반이 처음 보는 고장", 0.5, 0.6),
                                                     ("U3 고장 세기 무작위(×0.4~1.0) + 절반이 처음 보는 고장 (현실 조건)", 0.5, "rand")]):
        specs = make_unseen(make_specs(a.seed + 1000 + li * 5, a.test_scale, "test", sp["test"]), rng, share)
        for s in specs:                                     # 고장 세기 (약한 고장 조건, rand = 로봇마다 0.4~1.0배)
            if s["fault"]:
                strength_ = float(rng.uniform(0.4, 1.0)) if strength == "rand" else strength
                s["strength"] *= strength_
                if s.get("second"):
                    s["second"]["strength"] *= strength_
        tc = cache_of(specs, sp["test"])
        res["levels"][lname] = {}
        for k in keys:
            g = pick[k]
            v = evaluate(tc, specs, k, g["q"], g["cf"], g["cu"], classes)
            res["levels"][lname][k] = v
            log(f"{lname} · {k}: {v}")
        (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    names = {"ae_wheel": "특징 오토인코더 + 바퀴", "cnn_ae": "1D-CNN 오토인코더", "deep": "특징 AI + 1D-CNN (원래 주 모델)",
             "rf": "랜덤포레스트", "deep_rf": "특징 AI + 1D-CNN 오토인코더 + 랜덤포레스트", 
             "feat_rf": "특징 오토인코더 + 랜덤포레스트", "spec_ae": "주파수 모양 오토인코더", "spec_wheel": "주파수 모양 오토인코더 + 바퀴 점수",
             "spec_rf": "주파수 모양 오토인코더 + 랜덤포레스트 (새 주 모델, AI 2개)"}
    for lname, v in res["levels"].items():
        print(f"[{lname}]")
        for k, x in v.items():
            print(f"   {names.get(k, k)}: 정답률 {x['accuracy']}% · 아는 고장 감지 {x['known']}% · 처음 보는 고장 감지 {x['unseen']}% · "
                  f"오경보 {x['false_alarm']}% · 원인 {x['cause']}% ({x['n']}) {x['by_unseen']}")
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
