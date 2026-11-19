"""[역할] 실측 고장 기록(CSV 한 파일)에 '언제부터 고장인지' 라벨 열을 붙인다 (11/20)

    python 실행/라벨_붙이기.py --file data/실측_고장/run01.csv --start 6:10 --kind unbalance --wheel fr

  --start  고장을 낸 시각 (기록 시작부터 분:초 또는 초). 이 시각 전 = 정상, 이후 = 고장
  --kind   unbalance(바퀴 불균형 · 테이프) / looseness(체결 풀림) / bearing(마찰 증가 · 걸림) / 처음 보는 고장이면 unknown
  --wheel  fl(앞 왼쪽) · fr(앞 오른쪽) · rl(뒤 왼쪽) · rr(뒤 오른쪽)
결과: 같은 폴더에 원래 이름 + _label.csv (원래 파일은 그대로). fault_kind · fault_wheel · fault_severity 열이 붙음
시각 열은 팀 기록의 t_us(마이크로초)나 time(초)을 읽음.
"""
import argparse
from pathlib import Path

import pandas as pd


def seconds(text):
    """'6:10' → 370, '370' → 370"""
    if ":" in str(text):
        m, s = str(text).split(":", 1)
        return int(m) * 60 + float(s)
    return float(text)


def main():
    p = argparse.ArgumentParser(description="실측 고장 기록에 라벨 열 붙이기")
    p.add_argument("--file", required=True)
    p.add_argument("--start", required=True, help="고장 시작 시각 (분:초 또는 초)")
    p.add_argument("--kind", required=True, choices=["unbalance", "looseness", "bearing", "unknown"])
    p.add_argument("--wheel", required=True, choices=["fl", "fr", "rl", "rr"])
    p.add_argument("--severity", type=float, default=1.0, help="고장 정도 0~1 (기본 1)")
    a = p.parse_args()
    f = Path(a.file)
    df = pd.read_csv(f)
    if "t_us" in df.columns:
        t = (df["t_us"] - df["t_us"].iloc[0]) / 1e6
    elif "time" in df.columns:
        t = df["time"] - df["time"].iloc[0]
    else:
        raise SystemExit("시각 열(t_us 또는 time)이 없습니다.")
    start = seconds(a.start)
    if not 0 < start < float(t.iloc[-1]):
        raise SystemExit(f"--start {a.start}가 기록 길이({t.iloc[-1] / 60:.1f}분) 밖입니다.")
    after = t >= start
    df["fault_kind"] = ["normal" if not x else a.kind for x in after]
    df["fault_wheel"] = ["" if not x else a.wheel for x in after]
    df["fault_severity"] = [0.0 if not x else a.severity for x in after]
    if "label" in df.columns:                              # 팀 기록의 label 열(error 등)은 헷갈리지 않게 이름만 바꿔 둠
        df = df.rename(columns={"label": "label_원래"})
    out = f.with_name(f.stem + "_label.csv")
    df.to_csv(out, index=False)
    print(f"저장: {out} · 정상 {t.iloc[-1] if not after.any() else start:.0f}초 + 고장 {float(t.iloc[-1]) - start:.0f}초 ({a.kind}, {a.wheel})")
    if start < 360:
        print("주의: 고장 전 정상 구간이 6분보다 짧습니다. 처음 5분이 그 로봇의 '평소' 기준이라 6분 이상이 좋습니다.")


if __name__ == "__main__":
    main()
