"""실측 고장 기록을 바탕으로 만든 가상 로봇으로 학습 → 처음 보는 로봇 시험 (고장 감지율 %)

실측 고장 기록은 20초짜리 몇 개뿐이라, 우리 모델이 기대하는 '처음 5분 정상 → 고장' 형태의 긴 기록이 없다.
그래서 실측 조각을 이어 붙여 가상 로봇 기록을 만든다.
  - 정상 구간: 실측 정상 기록에서 2~6초 조각을 무작위로 골라 이어 붙임 (로봇마다 축별 크기 ±10%, 조각마다 ±5%, 작은 잡음)
  - 고장 구간: 실측 고장 기록에서 같은 방식으로. 고장 정도 a < 1은 정상 조각과 섞어 만듦: x = √(1-a)·정상 + √a·고장
    (진동 세기가 정상과 실측 고장 사이를 a만큼 옮겨 감. a = 1이 실측 고장 그대로)
  - 전류·전압·모터 명령·엔코더는 고른 조각의 값 그대로, 시간은 일정 간격으로 새로 매김
누출 방지: 겹마다 시험 로봇은 학습에 쓰지 않은 실측 파일(정상·고장 모두)로만 만든다.

실행 (second_model 폴더):
  python 가상테스트/실측기반_시험.py --normal "정상 데이터_욧골(11.15) 폴더" --fault "고장 데이터_욧골 폴더"
  기본: 속도 5 기록만 (같은 바닥·같은 속도의 정상이 있는 것), 고장 = 앞 오른쪽 바퀴 불균형(테이프)
결과: 가상테스트_결과/실측기반/ (GitHub에 안 올라감). 코드 점검·가능성 확인용이며 실측 성능이 아님.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config as C  # noqa: E402
from pipeline import dataio as D, workflow as W  # noqa: E402

VIB = ["ax", "ay", "az"]
KEEP = ["mV", "mA", "spd", "enc2"]
DT_US = 2080                                       # 실측 샘플 간격 중앙값 (약 480 Hz)


def load(folder, speed, skip):
    out = {}
    for f in sorted(Path(folder).glob("*.csv")):
        if any(k in f.name for k in skip):
            continue
        d = pd.read_csv(f)
        if speed is not None and not (d["spd"] == speed).all():
            continue
        if d["spd"].abs().max() == 0 or len(d) < 500:      # 멈춘 기록·너무 짧은 기록은 뺌
            continue
        out[f.stem] = d[VIB + KEEP].to_numpy(float)
    return out


class Stream:
    """실측 파일 묶음 → 원하는 길이만큼 조각을 이어 붙인 신호"""

    def __init__(self, files, rng, gain):
        self.files, self.rng, self.gain = list(files.values()), rng, gain

    def take(self, n):
        parts, got = [], 0
        while got < n:
            src = self.files[self.rng.integers(len(self.files))]
            L = int(self.rng.uniform(2, 6) / (DT_US * 1e-6))
            L = min(L, len(src))
            s = self.rng.integers(0, len(src) - L + 1)
            p = src[s:s + L].copy()
            p[:, :3] *= self.gain * self.rng.uniform(0.95, 1.05)
            if parts:                                       # 이음매 20샘플을 부드럽게 (튀는 값 방지)
                k = min(20, len(p), len(parts[-1]))
                w = np.linspace(0, 1, k)[:, None]
                p[:k, :3] = parts[-1][-k:, :3] * (1 - w) + p[:k, :3] * w
            parts.append(p)
            got += L
        return np.concatenate(parts)[:n]


def make_robot(rng, normals, faults, minutes, onset_min, sev):
    gain = rng.uniform(0.9, 1.1, 3)
    n_total = int(minutes * 60 / (DT_US * 1e-6))
    n_on = int(onset_min * 60 / (DT_US * 1e-6)) if sev > 0 else n_total
    ns = Stream(normals, rng, gain)
    x = ns.take(n_total)
    if sev > 0:
        fs = Stream(faults, rng, gain)
        f = fs.take(n_total - n_on)
        mix = x[n_on:].copy()
        mix[:, :3] = np.sqrt(1 - sev) * mix[:, :3] + np.sqrt(sev) * f[:, :3]
        mix[:, 3:] = f[:, 3:] if sev >= 0.5 else mix[:, 3:]   # 전류 등은 더 가까운 쪽 조각 값
        x[n_on:] = mix
    x[:, :3] += rng.normal(0, 40, (len(x), 3))               # 작은 잡음 (원래 정수값 40 ≈ 0.005 g)
    x[:, :3] = np.clip(np.round(x[:, :3]), -32768, 32767)    # 센서 한계(±4 g)도 실측처럼
    df = pd.DataFrame(x, columns=VIB + KEEP)
    df.insert(0, "t_us", 1_430_000 + DT_US * np.arange(len(df)) + rng.integers(-8, 9, len(df)))
    is_f = np.arange(len(df)) >= n_on
    df["fault_kind"] = np.where(is_f, "unbalance", "normal")
    df["fault_wheel"] = np.where(is_f, "fr", "")
    df["fault_severity"] = np.where(is_f, sev, 0.0)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--normal", required=True, help="실측 정상 CSV 폴더 (같은 바닥)")
    p.add_argument("--fault", required=True, help="실측 고장 CSV 폴더")
    p.add_argument("--speed", type=int, default=5, help="이 속도(spd)의 기록만 씀")
    p.add_argument("--skip", default="114650,114717", help="뺄 파일 이름 조각 (쉼표로). 기본: 바닥이 달라 보이는 욧골 정상 2개")
    p.add_argument("--minutes", type=float, default=12)
    p.add_argument("--onset", type=float, default=6, help="고장 시작(분). 처음 5분은 정상 기준")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "실측기반"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)
    skip = [k for k in a.skip.split(",") if k]
    normals, faults = load(a.normal, a.speed, skip), load(a.fault, a.speed, skip)
    fault_long = [k for k, v in faults.items() if len(v) >= 5000]
    log(f"실측 정상 {len(normals)}개 {list(normals)} · 실측 고장 {len(faults)}개 {list(faults)}")
    if len(fault_long) < 2 or len(normals) < 4:
        sys.exit("겹을 나누려면 긴 고장 기록 2개 이상, 정상 기록 4개 이상이 필요합니다.")

    # 겹: 긴 고장 기록마다 한 번씩 시험용으로 뺌. 정상 기록도 겹마다 나눔 (시험 ≈ 40%)
    nn = list(normals)
    rng0 = np.random.default_rng(a.seed)
    rows = []
    for i, test_fault in enumerate(fault_long):
        perm = list(rng0.permutation(nn))
        n_test = max(2, int(round(len(nn) * 0.4)))
        test_n, train_n = perm[:n_test], perm[n_test:]
        train_f = [k for k in faults if k != test_fault]
        rng = np.random.default_rng(a.seed + 100 * i)
        data = D.Dataset()
        specs = []
        # 학습: 정상 12대 + 고장 12대 (고장 정도 0.5·0.8·1.0 각 4대. 0.75~0.85 대역 = 위험도 100% 기준)
        for j in range(12):
            specs.append(("train", f"겹{i + 1}_학습_정상_{j + 1:02d}", 0.0, train_n, train_f))
        for j, sev in enumerate([0.5] * 4 + [0.8] * 4 + [1.0] * 4):
            specs.append(("train", f"겹{i + 1}_학습_고장{sev}_{j + 1:02d}", sev, train_n, train_f))
        # 시험: 정상 15대 + 고장 정도 1.0(실측 그대로)·0.5·0.25 각 10대
        for j in range(15):
            specs.append(("test", f"겹{i + 1}_시험_정상_{j + 1:02d}", 0.0, test_n, [test_fault]))
        for sev in [1.0, 0.5, 0.25]:
            for j in range(10):
                specs.append(("test", f"겹{i + 1}_시험_고장{sev}_{j + 1:02d}", sev, test_n, [test_fault]))
        for role, name, sev, nsrc, fsrc in specs:
            df = make_robot(rng, {k: normals[k] for k in nsrc}, {k: faults[k] for k in fsrc}, a.minutes, a.onset, sev)
            recs, fs = D._windows(df, name, data, None)
            data.robots[name], data.fs[name] = recs, fs
        test_names = [s[1] for s in specs if s[0] == "test"]
        log(f"겹 {i + 1}: 시험 고장 = {test_fault}, 시험 정상 = {test_n} / 학습 고장 = {train_f}, 학습 정상 = {train_n}")
        W.make_folds = lambda data_, names, t=test_names: [t]
        m = W.cross_validate(data, "per_unit", progress=lambda s: None)
        by = {u["robot"]: u for u in m["unit_results"]}
        for role, name, sev, _, _ in specs:
            if role != "test" or name not in by:
                continue
            u = by[name]
            first = C.minutes(u["caution"] + 1) if u["caution"] is not None else None
            rows.append(dict(겹=i + 1, 로봇=name, 고장정도=sev, 실제="고장" if sev > 0 else "정상",
                             첫경미_분=first, 첫심각_분=C.minutes(u["warning"] + 1) if u["warning"] is not None else None,
                             최고점수=u["max_score"], 시험고장파일=test_fault))
        aucs = {v["name"]: v["auc"] for v in m["detectors"].values()}
        log(f"겹 {i + 1} 끝 · 1초 창 AUC {aucs}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "로봇별_결과.csv", index=False, encoding="utf-8-sig")
    pct = lambda x, n: round(100 * x / n, 1) if n else None
    res = {}
    N = df[df["실제"] == "정상"]
    res["정상 오경보율"] = pct(int(N["첫경미_분"].notna().sum()), len(N))
    for sev in [1.0, 0.5, 0.25]:
        F = df[df["고장정도"] == sev]
        early = F["첫경미_분"].notna() & (F["첫경미_분"] <= a.onset)
        hit = F["첫경미_분"].notna() & (F["첫경미_분"] > a.onset)
        res[f"고장 {sev} 감지율"] = pct(int(hit.sum()), len(F))
        res[f"고장 {sev} 고장 전 경보"] = int(early.sum())
        res[f"고장 {sev} 평균 감지 시간(분)"] = round(float((F.loc[hit, "첫경미_분"] - a.onset).mean()), 1) if hit.any() else None
        res[f"고장 {sev} 심각까지 간 비율"] = pct(int(F["첫심각_분"].notna().sum()), len(F))
    for g, d in df.groupby("겹"):
        F, Nn = d[d["고장정도"] == 1.0], d[d["실제"] == "정상"]
        res[f"겹{g} 실측 그대로 감지율"] = pct(int((F["첫경미_분"] > a.onset).sum()), len(F))
        res[f"겹{g} 오경보율"] = pct(int(Nn["첫경미_분"].notna().sum()), len(Nn))
    (out / "결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "=" * 60)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    log(f"저장: {out}")


if __name__ == "__main__":
    main()
