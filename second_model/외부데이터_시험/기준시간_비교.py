# -*- coding: utf-8 -*-
"""정상 주행(기준) 시간 비교: 처음 정상 주행을 6분 → 4분 → 3분으로 줄이면 결과가 얼마나 달라지나 (외부 데이터, 동작 확인용)

같은 로봇 13대(같은 원본 기록, 같은 빠르기·크기 배율, 같은 고장 단계)에서 고장 로봇의 '처음 정상 주행'만 줄인 데이터를
    python 외부데이터_시험/만들기.py --normal-first 4 --out 외부데이터_시험/data_정상4분
    python 외부데이터_시험/만들기.py --normal-first 3 --out 외부데이터_시험/data_정상3분
로 만든 뒤, config.BASELINE_PERIODS(로봇마다 기준으로 쓰는 처음 몇 분)를 맞춰 로봇 단위 교차검증을 돌려 나란히 비교한다.
정상만 달리는 로봇 4대는 모든 경우에 같은 8분 기록을 쓴다.

[실행] python 외부데이터_시험/기준시간_비교.py   (약 15분) → 화면 표 + 가상테스트_결과/기준시간_비교.md
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import config as C  # noqa: E402
from pipeline import dataio as D, report, workflow as W  # noqa: E402
from pipeline.heartbeat import Heartbeat  # noqa: E402

# (이름, 데이터 폴더, 기준 분 = BASELINE_PERIODS)
CASES = [
    ("정상 6분 · 기준 5분 (지금 설정)", HERE / "data", 5),
    ("정상 4분 · 기준 3분", HERE / "data_정상4분", 3),
    ("정상 3분 · 기준 2분", HERE / "data_정상3분", 2),
    ("정상 3분 · 기준 5분 그대로 (설정 안 바꿈)", HERE / "data_정상3분", 5),
]
KEYS = ["판정 정답률", "고장 로봇 감지율", "정상 로봇 오경보율", "원인 분류 정확도", "이상 감지 AUC (오토인코더)",
        "고장 창 감지율 / 정상 창 오경보율"]


def main():
    missing = [str(p) for _, p, _ in CASES if not p.is_dir()]
    if missing:
        sys.exit("데이터 폴더가 없습니다 (만들기.py --normal-first 로 먼저 만드세요): " + ", ".join(missing))
    out_dir = HERE.parent / "가상테스트_결과"
    out_dir.mkdir(exist_ok=True)
    cache, results = {}, []
    with Heartbeat("기준 시간 비교 준비 중") as hb:
        for name, path, bp in CASES:
            C.BASELINE_PERIODS = bp                       # 이번 경우의 기준 시간 [분]
            if path not in cache:
                hb.set(f"데이터 읽는 중 ({path.name})")
                cache[path] = D.load(path)
            data = cache[path]
            mode = D.choose_baseline(data)                # auto: 기준을 못 만드는 로봇이 있으면 공통 기준(reference)으로 바뀜
            hb.set(f"평가 중: {name}")
            t0 = time.time()
            m = W.cross_validate(data, progress=lambda s: print(s, flush=True))
            head = dict((k, v) for k, v, _ in report.headline(m))
            miss = [r["로봇"] for r in report.unit_rows(m) if r["판정"] in ("놓침", "오경보")]
            results.append(dict(name=name, mode=mode, head=head, miss=miss, minutes=round((time.time() - t0) / 60, 1),
                                skipped=m["skipped"]))
            print(f"\n[{name}] 정상 기준 {mode} · " + " · ".join(f"{k} {head[k]}" for k in KEYS[:3]) + "\n", flush=True)

    lines = ["# 정상 주행(기준) 시간 비교 (외부 데이터, 동작 확인용 · 우리 로봇 성능 아님)", "",
             "같은 로봇 13대에서 고장 로봇의 처음 정상 주행 길이와 `BASELINE_PERIODS`(기준 분)만 바꿨습니다. 로봇 단위 5겹 교차검증.", "",
             "| 경우 | 정상 기준 방식 | " + " | ".join(KEYS) + " | 틀린 로봇 |", "|---|---|" + "---|" * len(KEYS) + "---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['mode']} | " + " | ".join(r["head"][k] for k in KEYS) + f" | {', '.join(r['miss']) or '없음'} |")
    for r in results:
        if r["skipped"]:
            lines += ["", f"- {r['name']}에서 평가에서 빠진 항목: " + "; ".join(r["skipped"])]
    (out_dir / "기준시간_비교.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("  정상 주행(기준) 시간 비교 (외부 데이터, 동작 확인용)")
    print("=" * 72)
    for r in results:
        print(f"  {r['name']}  [정상 기준 {r['mode']}]")
        for k in KEYS:
            print(f"      {k}: {r['head'][k]}")
        print(f"      틀린 로봇: {', '.join(r['miss']) or '없음'}")
    print("=" * 72)
    print("저장:", out_dir / "기준시간_비교.md")


if __name__ == "__main__":
    main()
