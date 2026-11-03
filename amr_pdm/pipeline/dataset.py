"""학습·평가용 데이터셋 생성: 여러 대의 가상 로봇을 주행시켜 특징 CSV로 저장

train_normal : 정상 로봇 6대 (이상 감지 모델 학습 + 점수 눈금 보정)
train_fault  : 고장 종류 3 × 바퀴 2 × 2대 = 12대 (원인 분류 학습 + 건강 점수 9 보정)
test         : 처음 보는 로봇 12대 (고장 6대는 진행 속도를 다르게, 정상 6대는 오경보 확인)
"""
import itertools
from multiprocessing import Pool

import pandas as pd

import config as C
from pipeline.features import extract
from sim.simulator import AMRSimulator


def scenarios():
    S = {"train_normal": [], "train_fault": [], "test": []}
    for i in range(6):
        S["train_normal"].append(dict(robot=f"N{i}", faults=[], n_laps=14, seed=100 + i))
    for i, (k, w, (s, e)) in enumerate(itertools.product(C.FAULT_KINDS, ["left", "right"], [(6, 30), (7, 26)])):
        S["train_fault"].append(dict(robot=f"F{i}", faults=[dict(kind=k, wheel=w, start_lap=s, end_lap=e)],
                                     n_laps=32, seed=200 + i))
    prog = [(8, 34), (10, 30), (12, 38), (9, 28), (11, 36), (8, 32)]
    for i, ((k, w), (s, e)) in enumerate(zip(itertools.product(C.FAULT_KINDS, ["right", "left"]), prog)):
        S["test"].append(dict(robot=f"T{i}", faults=[dict(kind=k, wheel=w, start_lap=s, end_lap=e)],
                              n_laps=40, seed=300 + i))
    for i in range(6):
        S["test"].append(dict(robot=f"TN{i}", faults=[], n_laps=40, seed=400 + i))
    return S


def describe(sc):
    if not sc["faults"]:
        return "정상"
    f = sc["faults"][0]
    return f"{C.SIDE_KO[f['wheel']]} 바퀴 {C.KIND_KO[f['kind']]} ({f['start_lap']}회차 시작 → {f['end_lap']}회차 최대)"


def run_one(sc):
    sim = AMRSimulator(faults=sc["faults"], n_laps=sc["n_laps"], seed=sc["seed"])
    rows = [extract(r) for r in sim.windows()]
    for r in rows:
        r["robot"], r["scenario"] = sc["robot"], describe(sc)
    return rows


def build(force=False, procs=4):
    C.DATA_DIR.mkdir(exist_ok=True)
    out = {}
    for split, scs in scenarios().items():
        path = C.DATA_DIR / f"features_{split}.csv.gz"
        if path.exists() and not force:
            out[split] = pd.read_csv(path)
            continue
        with Pool(procs) as p:
            rows = [r for chunk in p.map(run_one, scs) for r in chunk]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"  {split}: 로봇 {len(scs)}대, 창 {len(df):,}개 → {path.name}")
        out[split] = df
    return out
