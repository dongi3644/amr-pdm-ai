"""[역할] 실측 기록 + 가상 로봇을 함께 학습하고, 처음 보는 로봇으로 성능을 잰다 (11/19)

가정 (팀 결정, 11/19)
  - 모든 데이터에 고장 라벨이 있다 (실측 고장 = 앞 오른쪽 바퀴 테이프 = 바퀴 불균형)
  - 모든 데이터에 엔코더 4개가 있다. 실측은 엔코더가 앞 오른쪽(enc2) 하나뿐이라 나머지 3개를 가상으로 채운다
  - 실측 기록기에는 각속도가 없으므로 가상 로봇에서도 각속도 열을 빼고, gz(회전 속도)는 엔코더로 계산한다(dataio)
실측 로봇 만들기
  - 실측 파일은 20초씩이라 '처음 5분 정상 기준'이 없다 → 그 로봇의 실측 정상 파일을 섞어 되풀이한 5.5분을 앞에 붙이고,
    그 뒤에 실측 파일을 시간 순서대로 이어 붙임 (욧골: 정상 파일들 → 테이프 고장 파일들)
  - 앞 오른쪽 엔코더 = 실측 enc2를 누적 펄스로 바꾸고 바퀴 1회전 330펄스 눈금으로 맞춤 (끊김·0도 그대로 남음)
  - 나머지 3개 엔코더 = 모터 명령(spd)으로 정한 바퀴 속도 + 평소 흔들림 4.5%
  - 속도 5·7·정지 기록만 씀 (속도 3·9 등은 정상 기준이 없어서)
나누기 (겹치지 않음): 실측 파일은 RealPools.split으로 학습·검증·시험. 실측 테이프 고장은 학습 = s5_171612, 시험 = s5_172040·s5_172110.
  가상 불균형 고장의 모양(실측 증가분)도 학습용 고장 파일로만 잰다. 속도 7 테이프 고장은 같은 바닥 속도 7 정상이 없어 뺌
시험 조건 (단계마다 가상 시험 로봇을 새로 만듦, 실측 시험 로봇 2대는 같음)
  L0 기본 · L1 약한 고장(×0.6) · L2 L1 + 엔코더 끊김(로봇 40%) · L3 L2 + 더 약한 고장(×0.4) · L4 L3 + 처음 보는 베어링 주파수 절반
판정: 코드 기본 설정 그대로 (주 모델 = 오토인코더·랜덤포레스트·바퀴 점수 중 큰 값, 1분 점수 = 상위 25% 지점, 1번 만에, 경미 3점)
결과: 가상테스트_결과/실측혼합/ (GitHub에 없음)
"""
import argparse
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
from pipeline.models import NORMAL_LABEL  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402
import 실측보정_시험 as T  # noqa: E402
from 실측보정_난이도 import LEVELS, enc_glitch  # noqa: E402

GYRO = ["gx", "gy", "gz"]
ENC_K = None                                                # 실측 enc2(샘플마다 센 펄스) → 330펄스/회전 눈금 배율 (욧골 정상으로 잼)
TRAIN_FAULT, TEST_FAULT = ["vib_error_s5_171612"], ["vib_error_s5_172040", "vib_error_s5_172110"]


def load_real(folder, names):
    out = []
    for f in sorted(Path(folder).glob("*.csv")):
        if f.stem in names:
            out.append((f.stem, pd.read_csv(f)))
    return out


def real_part(df, rng, t0, kind, sev):
    """실측 CSV 1개 → 표준 열 표 (시간은 t0부터)"""
    n = len(df)
    t = (df["t_us"].to_numpy(float) - df["t_us"].iloc[0]) / 1e6 + t0
    spd = df["spd"].to_numpy(int)
    fw = R.FW_PER_SPD * spd
    out = pd.DataFrame({"time": t, "ax": df["ax"] / R.LSB, "ay": df["ay"] / R.LSB, "az": df["az"] / R.LSB,
                        "current": df["mA"] / 1000, "voltage": df["mV"] / 1000, "motor_cmd": spd})
    out["_fr"] = df["enc2"].to_numpy(float) * ENC_K         # 샘플마다 센 펄스 (눈금 맞춤) → 아래에서 누적
    dt = np.r_[np.diff(t), np.median(np.diff(t))]
    coarse = np.arange(t[0], t[-1] + 0.05, 0.05)
    for w in ("fl", "rl", "rr"):
        noise = np.interp(t, coarse, rng.standard_normal(len(coarse)))
        out[f"_{w}"] = fw * R.PPR * (1 + 0.045 * noise) * (spd > 0) * dt
    out["fault_kind"], out["fault_wheel"], out["fault_severity"] = kind, ("fr" if sev > 0 else ""), sev
    return out


def real_robot(name, floor, normals, faults, rng, base_min=5.5):
    """[그 로봇의 실측 정상 파일을 섞어 반복한 5.5분 = 정상 기준] + [실측 정상 파일들] + [실측 테이프 고장 파일들] → 로봇 1대의 표
    실측 파일이 20초씩이라 처음 5분 기준이 없으므로, 같은 로봇(같은 날·바닥)의 정상 파일을 되풀이해 채운다 (가상 조각을 쓰면 이음매가 고장처럼 보였음)"""
    parts, t0 = [], 0.0
    while t0 < base_min * 60:
        for i in rng.permutation(len(normals)):
            p = real_part(normals[i][1], rng, t0, "normal", 0.0)
            parts.append(p)
            t0 = float(p["time"].iloc[-1]) + 0.01
            if t0 >= base_min * 60:
                break
    for _, df in normals:
        p = real_part(df, rng, t0, "normal", 0.0)
        parts.append(p)
        t0 = float(p["time"].iloc[-1]) + 0.01
    for _, df in faults:
        p = real_part(df, rng, t0, "unbalance", 1.0)
        parts.append(p)
        t0 = float(p["time"].iloc[-1]) + 0.01
    full = pd.concat(parts, ignore_index=True)
    for w in ("fl", "fr", "rl", "rr"):                       # 엔코더: 누적 펄스 (16비트처럼 넘침)
        cnt = np.floor(np.cumsum(full.pop(f"_{w}").to_numpy()))
        full[f"enc_{w}"] = ((cnt.astype(np.int64) + 32768) % 65536) - 32768
    return full, len(normals), len(faults)


def main():
    global ENC_K
    p = argparse.ArgumentParser()
    p.add_argument("--g3", action="append", required=True)
    p.add_argument("--yg", action="append", required=True)
    p.add_argument("--fault", required=True)
    p.add_argument("--skip", default="114650,114717,110211")
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--test-scale", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--levels", default="0,1,2,3,4")
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "실측혼합"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJ), str(HERE)])
    skip = [k for k in a.skip.split(",") if k]

    pools = R.RealPools({"공3옥상": a.g3, "욧골": a.yg}, skip)
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fault_all = {f.stem: f for f in sorted(Path(a.fault).glob("*.csv"))}
    tf = [pd.read_csv(fault_all[k]) for k in TRAIN_FAULT]
    fm = R.FaultModel(sp["train"][("욧골", 5)], [np.column_stack([d.ax / R.LSB, d.ay / R.LSB, d.az / R.LSB, d.mA / 1000, d.mV / 1000]) for d in tf])
    # 실측 enc2 눈금: 욧골 정상(속도 5)의 샘플당 평균 펄스 → 바퀴 2.25 Hz × 330펄스
    yg_dir = Path(a.yg[0])
    e = np.mean([pd.read_csv(f)["enc2"].mean() for f in sorted(yg_dir.glob("*.csv")) if not any(k in f.name for k in skip)
                 and pd.read_csv(f)["spd"].mode()[0] == 5])
    ENC_K = R.FW_PER_SPD * 5 * R.PPR * (1 / R.FS) / e
    log(f"실측 enc2 눈금 배율 {ENC_K:.4f} (샘플당 {e:.2f}펄스 → 1회전 {R.PPR}펄스)")

    # 실측 로봇: 역할(학습·시험)마다 바닥별로 1대씩 (속도 5·7·정지 파일만, 시간 순서)
    real_specs = []
    folders = {"공3옥상": a.g3, "욧골": a.yg}
    rng = np.random.default_rng(a.seed + 77)
    data = D.Dataset()
    names_of = lambda role, floor: {n.split("/", 1)[1] for n in pools.names[role] if n.startswith(floor + "/")}
    for role in ("train", "test"):
        for floor in ("공3옥상", "욧골"):
            nn = names_of(role, floor)
            normals = [x for d in folders[floor] for x in load_real(d, nn)]
            normals = [x for x in normals if int(x[1]["spd"].mode()[0]) in (5, 7)]
            normals.sort(key=lambda x: x[0].split("_")[-1])
            faults = []
            if floor == "욧골":
                faults = [(k, pd.read_csv(fault_all[k])) for k in (TRAIN_FAULT if role == "train" else TEST_FAULT)]
            name = f"실측_{'학습' if role == 'train' else '시험'}_{floor}"
            table, n_n, n_f = real_robot(name, floor, normals, faults, rng)
            recs, fs = D._windows(table, name, data, None)
            data.robots[name], data.fs[name] = recs, fs
            real_specs.append(dict(name=name, role=role, fault=bool(n_f), floor=floor, n_normal=n_n, n_fault=n_f,
                                   onset=float(table.loc[table["fault_severity"] > 0, "time"].min() / 60) if n_f else None,
                                   minutes=round(float(table["time"].iloc[-1]) / 60, 1)))
            log(f"실측 로봇 {name}: 정상 파일 {n_n}개 + 고장 파일 {n_f}개 (앞 5.5분 = 이 로봇의 실측 정상 파일 반복), {real_specs[-1]['minutes']}분")

    train = T.make_specs(a.seed, a.scale, "train", sp["train"])
    for s in train:
        df = R.make_table(s, sp["train"], idle, fm).drop(columns=GYRO)
        recs, fs = D._windows(df, s["name"], data, None)
        data.robots[s["name"]], data.fs[s["name"]] = recs, fs
    train_names = [s["name"] for s in train] + [r["name"] for r in real_specs if r["role"] == "train"]
    pipe, notes = W.train(data, train_names, "per_unit")
    pipe.save(out / "model_bundle.joblib")
    b = pipe.bundle
    log(f"학습 끝: 가상 {len(train)}대 + 실측 2대 · 주 모델 {b['main']} · 특징 {len(b['feats'])}개 · 1분 점수 {b['period_quantile']} · {b['confirm_periods']}번")
    for nline in notes:
        log("  " + nline)

    from joblib import Parallel, delayed
    diag = lambda dd, specs: Parallel(n_jobs=a.jobs, backend="loky")(delayed(T._diag_one)(pipe, dd.robots[s["name"]], None, C.THRESH["caution"]) for s in specs)

    def score(specs, got):
        hit = fa = kind_ok = kind_n = 0
        nF = sum(s["fault"] for s in specs)
        rows = []
        for s, (first, lbl) in zip(specs, got):
            ok_hit = bool(s["fault"] and first is not None and first >= int(s["onset"] * 60 // C.PERIOD_S))
            if s["fault"]:
                hit += ok_hit
                kinds = {s.get("kind", "unbalance")} | ({s["second"]["kind"]} if s.get("second") else set())
                if lbl and lbl != NORMAL_LABEL:
                    kind_n += 1
                    kind_ok += lbl.split("|")[0] in kinds
            else:
                fa += first is not None
            rows.append(dict(name=s["name"], fault=s["fault"], first_min=C.minutes(first + 1) if first is not None else None,
                             cause=lbl, hit=ok_hit))
        nN = len(specs) - nF
        return dict(robots=len(specs), fault=nF, normal=nN,
                    accuracy=round(100 * (hit + nN - fa) / len(specs), 1),
                    detect=round(100 * hit / nF, 1) if nF else None, false_alarm=round(100 * fa / nN, 1) if nN else None,
                    kind_ok=round(100 * kind_ok / kind_n, 1) if kind_n else None,
                    hits=f"{hit}/{nF}", false=f"{fa}/{nN}", causes=f"{kind_ok}/{kind_n}"), rows

    res = dict(settings=dict(main=b["main"], feats=len(b["feats"]), period_quantile=b["period_quantile"],
                             confirm=b["confirm_periods"], thresh=C.THRESH, rf_detector=C.RF_DETECTOR, enc_k=ENC_K),
               real_robots=real_specs, notes=notes)
    rt = [s for s in real_specs if s["role"] == "test"]
    res["실측 시험 로봇"], res["실측 시험 로봇 상세"] = score(rt, diag(data, rt))
    log(f"실측 시험 로봇 2대: {res['실측 시험 로봇']} · {res['실측 시험 로봇 상세']}")

    for li in [int(x) for x in a.levels.split(",")]:
        lname, opt = LEVELS[li]
        rng = np.random.default_rng(a.seed + 500 + li)
        specs = T.make_specs(a.seed + 1000 + li, a.test_scale, "test", sp["test"])
        tdata = D.Dataset()
        n_gl = 0
        for s in specs:
            s["name"] = f"L{li}_{s['name']}"
            if s["fault"]:
                s["strength"] *= opt.get("strength", 1.0)
                if s.get("second"):
                    s["second"]["strength"] *= opt.get("strength", 1.0)
                if rng.random() < opt.get("unseen", 0.0):
                    s.update(kind="bearing", bearing_hz=float(rng.choice([120.0, 320.0])))
            df = R.make_table(s, sp["test"], idle, fm).drop(columns=GYRO)
            if rng.random() < opt.get("enc_glitch", 0.0):
                enc_glitch(df, rng)
                n_gl += 1
            recs, fs = D._windows(df, s["name"], tdata, None)
            tdata.robots[s["name"]], tdata.fs[s["name"]] = recs, fs
        sm, rows = score(specs, diag(tdata, specs))
        res[lname] = dict(sm, enc_glitch_robots=n_gl, condition=opt)
        res[lname + " 로봇별"] = rows
        log(f"{lname}: {sm} (엔코더 끊김 {n_gl}대)")
        (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"설정: 주 모델 {b['main']} · 특징 {len(b['feats'])}개 · 1분 점수 {b['period_quantile']} · {b['confirm_periods']}번 · 경미 {C.THRESH['caution']}점")
    v = res["실측 시험 로봇"]
    print(f"[실측 시험 로봇 2대] 감지 {v['hits']} · 오경보 {v['false']} · 원인 {v['causes']}")
    for li in [int(x) for x in a.levels.split(",")]:
        v = res[LEVELS[li][0]]
        print(f"[{LEVELS[li][0]}] {v['robots']}대: 정답률 {v['accuracy']}% · 감지 {v['detect']}% ({v['hits']}) · 오경보 {v['false_alarm']}% ({v['false']}) · 원인 {v['kind_ok']}% ({v['causes']})")
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
