"""[역할] 실측 데이터로 보정한 가상 로봇(실측보정_로봇.py)으로 학습 → 검증 로봇으로 설정 고르기 → 처음 보는 시험 로봇으로 최종 성능 (11/18)

    python 가상테스트/실측보정_시험.py --g3 "정상 데이터_공3옥상" --g3 "정상 데이터_공3옥상(11.16)" \
        --yg "정상 데이터_욧골(11.15)" --fault "고장 데이터_욧골" [--scale 0.6]

순서
  1) 실측 파일을 학습·검증·시험용으로 나눔 (파일 단위, 겹치지 않음). 불균형 고장 모양은 학습용 욧골 정상 vs 실측 테이프 고장으로 잼
  2) 가상 로봇: 학습 / 검증 / 시험 (각자 자기 몫의 실측 조각으로만 만듦). 엔코더 4개·각속도는 항상 있음
  3) 학습: 지금 파이프라인(오토인코더 + 바퀴 점수 + 원인 랜덤포레스트) + 비교용 랜덤포레스트 감지기(compare_ae_rf.add_rf)
  4) 검증 로봇으로 '이상 점수 방법 × 주기 점수 지점 × 연속 주기 × 경미 기준' 조합을 모두 채점 → 가장 좋은 조합 선택
     (점수 = (고장 감지율 + (100 - 오경보율)) / 2, 오경보율 5% 이하만)
  5) 시험 로봇(처음 보는 실측 조각)으로 고른 조합과 지금 기본 설정을 채점. 고른 조합은 실제 진단 코드(workflow.diagnose)로도 다시 확인
결과: 가상테스트_결과/실측보정/ (GitHub에 안 올라감). 실측 조각 + 가정으로 만든 가상 로봇이라 실측 성능이 아님.
"""
import argparse
import itertools
import json
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
from pipeline.models import NORMAL_LABEL, health  # noqa: E402
import sim_robots as S  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402
from compare_ae_rf import add_rf  # noqa: E402

SETUPS = {"ae_wheel": "오토인코더 + 바퀴 점수 (지금)", "ae": "오토인코더만", "rf": "랜덤포레스트만",
          "rf_wheel": "랜덤포레스트 + 바퀴 점수", "ae_rf": "오토인코더·랜덤포레스트 중 큰 쪽",
          "ae_rf_wheel": "오토인코더·랜덤포레스트 중 큰 쪽 + 바퀴 점수"}
GRID = dict(quantile=[0.5, 0.75], confirm=[1, 2, 3], caution=[2.0, 2.5, 3.0])
DEFAULT = ("ae_wheel", 0.5, 2, 3.0)

TRAIN = {"정상": 18, "바닥 바뀜": 5, "짐 실음": 4, "저전압 정상": 3, "속도 바뀜": 3,
         "급성 고장": 6, "간헐 고장": 6, "복합 고장": 8, "약한 고장": 6, "속도 다름": 4}
TRAIN_GRADUAL_EACH = 2
EVAL_COMPLEX = {"급성 고장": 5, "간헐 고장": 5, "복합 고장": 5, "약한 고장": 5, "속도 다름": 5, "베어링 변형": 5, "고장+저전압": 5}
EVAL_HARD = {"바닥 바뀜": 6, "짐 실음": 4, "저전압 정상": 4, "속도 바뀜": 4}
EVAL_GENERAL = 50


def place(spec, rng, pools, role, cat):
    """로봇에 바닥·속도를 정함 (그 역할 몫의 실측 묶음 안에서)"""
    have = lambda k: k in pools and len(pools[k]) > 0
    yg = have(("욧골", 5))
    if cat == "바닥 바뀜":
        spec.update(floor="공3옥상", spd=5, floor2="욧골" if yg else "공3옥상")
        return
    if cat == "속도 바뀜":
        spec.update(floor="공3옥상", spd=5, spd2=7, spd_at=rng.uniform(7, 10))
        return
    if cat == "속도 다름" and role != "train":            # 시험·검증: 학습에 없던 속도 (실측 속도 3·9 묶음이 있으면)
        cand = [k for k in [("공3옥상", 9), ("공3옥상", 3)] if have(k)]
        if cand:
            spec.update(floor=cand[rng.integers(len(cand))][0], spd=cand[rng.integers(len(cand))][1])
            if not have((spec["floor"], spec["spd"])):
                spec.update(floor=cand[0][0], spd=cand[0][1])
            return
    if yg and rng.random() < 0.35:
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
                specs += [S.complex_spec(rng, f"학습_점진_{k}_{w}_{j + 1}", "점진 고장", k, w) for j in range(nn(TRAIN_GRADUAL_EACH))]
        for cat, cnt in TRAIN.items():
            specs += [S.complex_spec(rng, f"학습_{cat.replace(' ', '')}_{j + 1:02d}", cat if cat != "속도 바뀜" else "정상")
                      | {"cat": cat} for j in range(nn(cnt))]
    else:
        tag = "검증" if role == "val" else "시험"
        for i in range(nn(EVAL_GENERAL)):
            s = S.random_spec(rng, f"{tag}_일반{i + 1:03d}", rng.random() < 0.5)
            s.update(cat="일반 고장" if s["fault"] else "일반 정상", group="일반")
            specs.append(s)
        for cat, cnt in EVAL_COMPLEX.items():
            specs += [S.complex_spec(rng, f"{tag}_복잡_{cat.replace(' ', '')}_{j + 1:02d}", cat, extreme=True) | {"group": "복잡한 고장"}
                      for j in range(nn(cnt))]
        for cat, cnt in EVAL_HARD.items():
            specs += [S.complex_spec(rng, f"{tag}_어려운정상_{cat.replace(' ', '')}_{j + 1:02d}", cat if cat != "속도 바뀜" else "정상", extreme=True)
                      | {"group": "어려운 정상", "cat": cat} for j in range(nn(cnt))]
    for s in specs:
        s.setdefault("group", "학습")
        place(s, rng, pools, role, s["cat"])
        if s["cat"] == "바닥 바뀜" and s.get("floor_at") is None:
            s["floor_at"] = rng.uniform(7, 10)
    return specs


def fast_status(periods, scores, q, confirm, caution):
    """주기마다 점수 지점 → 단계 (2번 연속 규칙). workflow.diagnose의 판정을 빠르게 흉내 냄 (설정 고르기용)"""
    out, state, cand, run = [], 0, 0, 0
    for p in sorted(set(periods)):
        if p < C.BASELINE_PERIODS:
            continue
        v = float(np.quantile(scores[periods == p], q))
        lvl = 2 if v >= C.THRESH["warning"] else (1 if v >= caution else 0)
        if lvl == cand:
            run += 1
        else:
            cand, run = lvl, 1
        if run >= confirm and lvl != state:
            state = lvl
        out.append((p, state))
    return out


def score_robots(cache, specs, key, q, confirm, caution):
    rows = []
    for s in specs:
        per, sc = cache[s["name"]]["period"], cache[s["name"]][key]
        st = fast_status(per, sc, q, confirm, caution)
        first = next((p for p, x in st if x >= 1), None)
        onset_p = int(s["onset"] * 60 // C.PERIOD_S) if s["fault"] else None
        rows.append(dict(name=s["name"], cat=s["cat"], group=s["group"], fault=bool(s["fault"]), first=first,
                         hit=bool(s["fault"] and first is not None and first >= onset_p),
                         early=bool(s["fault"] and first is not None and first < onset_p),
                         fa=bool(not s["fault"] and first is not None),
                         delay=(C.minutes(first + 1) - s["onset"]) if (s["fault"] and first is not None and first >= onset_p) else None))
    return rows


def summary(rows):
    pc = lambda a, b: round(100 * a / b, 1) if b else None
    F = [r for r in rows if r["fault"]]
    N = [r for r in rows if not r["fault"]]
    det, fa = pc(sum(r["hit"] for r in F), len(F)), pc(sum(r["fa"] for r in N), len(N))
    acc = pc(sum(r["hit"] for r in F) + sum(not r["fa"] for r in N), len(rows))
    d = [r["delay"] for r in F if r["delay"] is not None]
    return dict(robots=len(rows), fault=len(F), normal=len(N), detect=det, false_alarm=fa, accuracy=acc,
                balanced=round((det + 100 - fa) / 2, 1) if det is not None and fa is not None else None,
                early=sum(r["early"] for r in F), delay=round(float(np.mean(d)), 1) if d else None)


def main():
    p = argparse.ArgumentParser(description="실측 보정 가상 로봇 학습·검증·시험")
    p.add_argument("--g3", action="append", required=True, help="공3옥상 정상 폴더 (여러 번)")
    p.add_argument("--yg", action="append", required=True, help="욧골 정상 폴더")
    p.add_argument("--fault", required=True, help="실측 고장(테이프, 앞 오른쪽) 폴더")
    p.add_argument("--skip", default="114650,114717,110211", help="뺄 파일 이름 조각")
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-fa", type=float, default=5.0, help="검증에서 허용하는 최대 오경보율 [%%]")
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "실측보정"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    skip = [k for k in a.skip.split(",") if k]

    pools = R.RealPools({"공3옥상": a.g3, "욧골": a.yg}, skip)
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fp = R.RealPools({"욧골": [a.fault]})
    fm = R.FaultModel(sp["train"][("욧골", 5)], [x for _, x in fp.files[("욧골", 5)]])
    log("실측 파일 나누기: " + " / ".join(f"{r} {len(v)}개" for r, v in pools.names.items())
        + f" · 불균형 고장 증가분(분산 g²) {[round(v, 3) for v in fm.var]}")
    (out / "실측파일_나누기.json").write_text(json.dumps(pools.names, ensure_ascii=False, indent=1), encoding="utf-8")

    specs = {r: make_specs(a.seed + i, a.scale, r, sp[r]) for i, r in enumerate(["train", "val", "test"])}
    data = D.Dataset()
    for r, lst in specs.items():
        for s in lst:
            recs, fs = D._windows(R.make_table(s, sp[r], idle, fm), s["name"], data, None)
            data.robots[s["name"]], data.fs[s["name"]] = recs, fs
        log(f"가상 로봇 {r} {len(lst)}대 만듦 (고장 {sum(s['fault'] for s in lst)}대)")
    train_names = [s["name"] for s in specs["train"]]
    pipe, notes = W.train(data, train_names, "per_unit")
    n_rf = add_rf(pipe, data, train_names)
    log(f"학습 끝 (창 {n_rf:,}개). 감지기: {list(pipe.bundle['detectors'])}")
    keys = [k for k in SETUPS if k in pipe.bundle["detectors"]]

    cache = {}
    for r in ("val", "test"):
        for s in specs[r]:
            zrows, Z = pipe.unit_z(W.rows_of(data.robots[s["name"]], s["name"]))
            c = {"period": np.array([z["period"] for z in zrows])}
            for k in keys:
                c[k] = health(pipe.bundle["detectors"][k], Z)
            cache[s["name"]] = c
    log("검증·시험 로봇 점수 계산 끝")

    # 4) 검증으로 조합 고르기
    grid = []
    for k, q, cf, cu in itertools.product(keys, GRID["quantile"], GRID["confirm"], GRID["caution"]):
        sm = summary(score_robots(cache, specs["val"], k, q, cf, cu))
        grid.append(dict(setup=k, quantile=q, confirm=cf, caution=cu, **sm))
    ok = [g for g in grid if g["false_alarm"] <= a.max_fa] or grid
    best = max(ok, key=lambda g: (g["balanced"], -g["false_alarm"], -(g["delay"] or 99)))
    grid.sort(key=lambda g: -g["balanced"])
    log(f"검증 최고: {best}")

    # 5) 시험
    pick = (best["setup"], best["quantile"], best["confirm"], best["caution"])
    res = {}
    for label, (k, q, cf, cu) in [("기본 설정", DEFAULT), ("고른 설정", pick)]:
        rows = score_robots(cache, specs["test"], k, q, cf, cu)
        res[label] = dict(setup=k, name=SETUPS[k], quantile=q, confirm=cf, caution=cu, all=summary(rows),
                          groups={g: summary([x for x in rows if x["group"] == g]) for g in ["일반", "복잡한 고장", "어려운 정상"]},
                          cats={c: summary([x for x in rows if x["cat"] == c]) for c in dict.fromkeys(x["cat"] for x in rows)},
                          kinds={kk: summary([x for x, s in zip(rows, specs["test"]) if s["fault"] and s["kind"] == kk]) for kk in S.KINDS})
    for label in ("기본 설정", "고른 설정"):
        res[label]["by_setup_test"] = None
    res["모든 방법 시험 (각 방법의 검증 최고 조합)"] = {}
    for k in keys:
        cand = [g for g in ok if g["setup"] == k] or [g for g in grid if g["setup"] == k]
        b = max(cand, key=lambda g: (g["balanced"], -g["false_alarm"]))
        res["모든 방법 시험 (각 방법의 검증 최고 조합)"][SETUPS[k]] = dict(
            quantile=b["quantile"], confirm=b["confirm"], caution=b["caution"],
            **summary(score_robots(cache, specs["test"], k, b["quantile"], b["confirm"], b["caution"])))

    # 실제 진단 코드(workflow.diagnose)로 다시 채점 (원인 정확도 포함). 주기 점수 지점·연속 횟수는 모델(bundle)에 저장된 값을 쓰므로 거기에 넣음
    def real_check(k, q, cf, cu):
        pipe.bundle["period_quantile"], pipe.bundle["confirm_periods"] = q, cf
        C.THRESH = dict(C.THRESH, caution=cu)
        hit = fa = nF = nN = kind_ok = kind_n = 0
        for s in specs["test"]:
            table, mon = W.diagnose(pipe, data.robots[s["name"]], detector=k)
            first = next((pp for pp, st in mon.events if st >= 1), None)
            if s["fault"]:
                nF += 1
                hit += first is not None and first >= int(s["onset"] * 60 // C.PERIOD_S)
                lbl = mon.last_cause_label
                kinds = {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())
                if lbl and lbl != NORMAL_LABEL:
                    kind_n += 1
                    kind_ok += lbl.split("|")[0] in kinds
            else:
                nN += 1
                fa += first is not None
        C.THRESH = dict(C.THRESH, caution=3.0)
        return dict(setup=SETUPS[k], quantile=q, confirm=cf, caution=cu, detect=round(100 * hit / nF, 1),
                    false_alarm=round(100 * fa / nN, 1), kind_ok=round(100 * kind_ok / kind_n, 1) if kind_n else None, kind_n=kind_n)

    best_of = {k: max([g for g in ok if g["setup"] == k] or [g for g in grid if g["setup"] == k],
                      key=lambda g: (g["balanced"], -g["false_alarm"])) for k in keys}
    checks = [DEFAULT] + [(k, best_of[k]["quantile"], best_of[k]["confirm"], best_of[k]["caution"]) for k in ("ae_wheel", "rf_wheel", "ae_rf_wheel") if k in best_of]
    res["실제 진단 코드 시험"] = [real_check(*c) for c in dict.fromkeys(checks)]
    for v in res["실제 진단 코드 시험"]:
        log(f"실제 진단: {v}")
    pipe.bundle["period_quantile"], pipe.bundle["confirm_periods"] = C.PERIOD_QUANTILE, C.CONFIRM_PERIODS
    pipe.save(out / "model_bundle.joblib")
    res["검증 상위 10"] = grid[:10]
    res["학습"] = dict(robots=len(specs["train"]), notes=notes)
    (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    for label in ("기본 설정", "고른 설정"):
        v = res[label]
        print(f"[{label}] {v['name']} · 주기 점수 {v['quantile']} · {v['confirm']}번 연속 · 경미 기준 {v['caution']}점")
        print(f"   시험 {v['all']['robots']}대: 고장 감지율 {v['all']['detect']}% · 오경보율 {v['all']['false_alarm']}% · 정답률 {v['all']['accuracy']}%"
              f" · 평균 감지 {v['all']['delay']}분")
        for g, x in v["groups"].items():
            print(f"     {g}: 감지 {x['detect']}% · 오경보 {x['false_alarm']}% ({x['robots']}대)")
        print("     종류별 감지: " + ", ".join(f"{C.KIND_KO[k]} {x['detect']}%" for k, x in v["kinds"].items()))
    print("[실제 진단 코드]")
    for v in res["실제 진단 코드 시험"]:
        print(f"   {v['setup']} (주기 {v['quantile']}, {v['confirm']}번, 기준 {v['caution']}): 감지 {v['detect']}% · 오경보 {v['false_alarm']}% · 원인 맞음 {v['kind_ok']}% ({v['kind_n']}대)")
    print("[모든 방법]")
    for k, v in res["모든 방법 시험 (각 방법의 검증 최고 조합)"].items():
        print(f"   {k}: 감지 {v['detect']}% · 오경보 {v['false_alarm']}% (주기 {v['quantile']}, {v['confirm']}번, 기준 {v['caution']})")
    print("=" * 70)
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
