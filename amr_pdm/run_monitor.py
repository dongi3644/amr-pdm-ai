"""저장된 파이프라인으로 처음 보는 로봇 1대를 실시간처럼 진단해 회차별로 출력한다.

    python run_monitor.py                    # 우측 바퀴 베어링 결함
    python run_monitor.py looseness left     # 고장 종류·바퀴 지정
    python run_monitor.py normal             # 정상 로봇
"""
import sys

import config as C
from pipeline.system import AMRPipeline
from sim.simulator import AMRSimulator


def main(kind="bearing", wheel="right"):
    faults = [] if kind == "normal" else [dict(kind=kind, wheel=wheel, start_lap=8, end_lap=32)]
    sim = AMRSimulator(faults=faults, n_laps=36, seed=500)
    monitor = AMRPipeline.load().monitor()
    print("회차 | 실제 고장 | 점수 | 상태 | 남은 회차(90% 범위) | 추정 원인 | 판단 근거")
    shown, sev, out = 0, {}, None

    def show():
        nonlocal shown
        for lap, score in monitor.lap_scores[shown:]:
            fc = monitor.forecast
            eta = "-" if not fc else f"{fc['laps']:.1f} ({fc['laps_lo']:.1f}~{'' if fc['laps_hi'] == float('inf') else format(fc['laps_hi'], '.1f')})"
            why = ", ".join(name for name, _ in monitor.reason) or "-"
            print(f"{lap:3d}  | {sev[lap]:.2f}      | {score:4.1f} | {C.STATUS_KO[monitor.status]} | {eta} | "
                  f"{(out and out['cause']) or '-'} | {why}")
        shown = len(monitor.lap_scores)

    for rec in sim.windows():
        res = monitor.update(rec)
        sev[res["lap"]] = res["row"]["true_sev"]
        show()
        if res["phase"] == "monitor":
            out = res
    monitor.finish()
    show()
    print("\n상태 변경 이력:", [(lap, C.STATUS_KO[s]) for lap, s in monitor.events] or "없음")


if __name__ == "__main__":
    main(*sys.argv[1:3])
