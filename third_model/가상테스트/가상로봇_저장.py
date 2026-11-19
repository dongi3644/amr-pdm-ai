"""[역할] 가상 로봇 몇 대만 골라 CSV 파일로 저장 (열어 보기용, 11/20)

성능 시험(성능시험.py)은 가상 로봇 수백 대를 메모리에서만 만들고 버린다. 이 스크립트는 같은 생성기(실측보정_로봇.py)로
원하는 로봇 몇 대만 만들어 CSV로 남긴다. 저장한 CSV는 `python run.py check · diagnose --path 저장폴더`로도 읽힌다.

    python 가상테스트/가상로봇_저장.py                                   # 정상 1대 + 고장 3종류(불균형 · 풀림 · 마찰) 1대씩
    python 가상테스트/가상로봇_저장.py --normal 2 --fault unbalance,flat --wheel rr --strength 0.6
    python 가상테스트/가상로봇_저장.py --fault gear --hz 50              # 엑셀로 보기 좋게 1초에 50줄로 줄여 저장 (진단에는 쓰지 말 것)

  --normal N     정상 로봇 N대 (기본 1)
  --fault 목록   고장 로봇 종류 (쉼표로): unbalance(바퀴 불균형 · 테이프) · looseness(체결 풀림) · bearing(마찰 증가)
                 · gear(기어 마모, 시험 전용 '처음 보는 고장') · flat(바퀴 찍힘, 시험 전용). 빈칸이면 고장 로봇 없음
  --each N       고장 종류마다 몇 대 (기본 1)
  --wheel        고장 바퀴 fl · fr · rl · rr (기본 fr = 실측 테이프와 같은 앞 오른쪽)
  --strength     고장 세기 배수 (기본 1.0 = 실측 테이프 수준. 0.6이면 약한 고장)
  --floor --spd  바닥(공3옥상 · 욧골) · 속도 단계(5 · 7). 기본 공3옥상 · 5
  --hz           저장 샘플링 [Hz]. 기본 = 원래 그대로(약 480 Hz, 10분에 약 30 MB). 줄이면 파일이 작아지지만 고주파가 사라져 진단용이 아님
  실측 재료 폴더: --g3 · --yg · --fault-dir (기본 data/ 아래 팀 실측 폴더 4개, data/README.md)
결과: --out 폴더(기본 가상테스트_결과/가상로봇_CSV)에 로봇마다 CSV 1개 + 목록.txt (종류 · 바퀴 · 고장 시작 분 · 세기 · 바닥 · 속도 · 길이, 탭 구분 · 엑셀로 열림)
열: time[s], ax · ay · az[g], current[A], voltage[V], motor_cmd, enc_fl · enc_fr · enc_rl · enc_rr(누적 펄스), fault_kind · fault_wheel · fault_severity
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
import sim_robots as S  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402

KIND_KO = {"normal": "정상", "unbalance": "바퀴불균형", "looseness": "체결풀림", "bearing": "마찰증가", "gear": "기어마모", "flat": "바퀴찍힘"}
COLS = ["time", "ax", "ay", "az", "current", "voltage", "motor_cmd", "enc_fl", "enc_fr", "enc_rl", "enc_rr",
        "fault_kind", "fault_wheel", "fault_severity"]


def main():
    p = argparse.ArgumentParser(description="가상 로봇 몇 대를 CSV로 저장")
    p.add_argument("--normal", type=int, default=1)
    p.add_argument("--fault", default="unbalance,looseness,bearing")
    p.add_argument("--each", type=int, default=1)
    p.add_argument("--wheel", default="fr", choices=S.WHEELS)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--floor", default="공3옥상", choices=["공3옥상", "욧골"])
    p.add_argument("--spd", type=int, default=5, choices=[5, 7])
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--g3", action="append", default=None)
    p.add_argument("--yg", action="append", default=None)
    p.add_argument("--fault-dir", default=str(PROJ / "data" / "고장 데이터_욧골"))
    p.add_argument("--skip", default="114650,114717,110211")
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "가상로봇_CSV"))
    a = p.parse_args()
    g3 = a.g3 or [str(PROJ / "data" / "정상 데이터_공3옥상"), str(PROJ / "data" / "정상 데이터_공3옥상(11.16)")]
    yg = a.yg or [str(PROJ / "data" / "정상 데이터_욧골(11.15)")]
    for d in g3 + yg + [a.fault_dir]:
        if not Path(d).is_dir():
            raise SystemExit(f"실측 재료 폴더가 없습니다: {d}\n팀 실측 폴더 4개를 data/ 아래에 두거나 --g3 · --yg · --fault-dir로 알려 주세요 (data/README.md).")
    if a.floor == "욧골" and a.spd != 5:
        raise SystemExit("욧골 실측 정상은 속도 5만 있습니다 (--spd 5).")

    pools = R.RealPools({"공3옥상": g3, "욧골": yg}, [k for k in a.skip.split(",") if k])
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fp = R.RealPools({"욧골": [a.fault_dir]})
    fm = R.FaultModel(sp["train"][("욧골", 5)], [x for _, x in fp.files[("욧골", 5)]])
    use = sp["test"]                                       # 시험 몫의 실측 조각으로 만듦 (성능시험.py의 시험 로봇과 같은 재료)
    if (a.floor, a.spd) not in use:
        use = {**use, (a.floor, a.spd): sp["train"][(a.floor, a.spd)]}

    rng = np.random.default_rng(a.seed + 99)
    specs = []
    for i in range(a.normal):
        s = S.random_spec(rng, f"가상_정상_{i + 1:02d}", False)
        specs.append(s)
    for kind in [k.strip() for k in a.fault.split(",") if k.strip()]:
        if kind not in KIND_KO or kind == "normal":
            raise SystemExit(f"모르는 고장 종류: {kind} (unbalance · looseness · bearing · gear · flat)")
        for j in range(a.each):
            s = S.random_spec(rng, f"가상_{KIND_KO[kind]}_{a.wheel}_{j + 1:02d}", True,
                              kind="unbalance" if kind in ("gear", "flat") else kind, wheel=a.wheel)
            s["kind"] = kind
            s["strength"] *= a.strength
            specs.append(s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in specs:
        s.update(floor=a.floor, spd=a.spd)
        df = R.make_table(s, use, idle, fm)
        df = df[[c for c in COLS if c in df.columns]]
        if a.hz:                                           # 보기용으로 줄이기 (k줄마다 1줄)
            k = max(1, int(round(R.FS / a.hz)))
            df = df.iloc[::k]
        f = out / f"{s['name']}.csv"
        t_end = float(df["time"].iloc[-1])
        df = df.assign(time=df["time"].map(lambda v: f"{v:.5f}"))   # 시각은 소수 5자리까지 (유효숫자로 자르면 1000초 넘어 겹침)
        df.to_csv(f, index=False, float_format="%.6g")
        rows.append(dict(파일=f.name, 종류=KIND_KO[s["kind"]], 바퀴=s.get("wheel") or "-", 바닥=a.floor, 속도=a.spd,
                         길이_분=round(t_end / 60, 1),
                         고장_시작_분=round(s["onset"], 1) if s["fault"] else None,
                         세기=round(s["strength"], 2) if s["fault"] else None,
                         최종_고장정도=round(s["final"], 2) if s["fault"] else None,
                         샘플링_Hz=round(a.hz or R.FS), 줄_수=len(df), 크기_MB=round(f.stat().st_size / 1e6, 1)))
        print(f"저장: {f.name}  ({rows[-1]['길이_분']}분, {rows[-1]['크기_MB']} MB)")
    # 목록은 .txt(탭 구분)로: 폴더를 run.py check · diagnose --path로 읽을 때 데이터 파일(.csv)로 섞이지 않게. 엑셀에서 그대로 열림
    pd.DataFrame(rows).to_csv(out / "목록.txt", index=False, sep="	", encoding="utf-8-sig")
    print(f"목록: {out / '목록.txt'}")


if __name__ == "__main__":
    main()
