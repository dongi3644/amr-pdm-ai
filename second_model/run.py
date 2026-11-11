"""[역할] 명령줄 입구. 사용자가 고른 명령(check / train / diagnose)을 pipeline의 함수로 연결한다.
       계산은 하지 않고 '무엇을 어떤 순서로 부를지'와 '결과를 어디에 저장할지'만 정한다.

CSV를 data 폴더에 넣고 아래 순서로 실행한다. 형식은 README.md

    python run.py check                 # 1) 파일 형식 검사          dataio.load → 표·경고 출력
    python run.py train                 # 2) 평가 + 최종 학습         workflow.cross_validate → workflow.train → reports/, artifacts/
    python run.py diagnose              # 3) 로봇별 진단 (분 단위)    저장된 모델 → workflow.diagnose → reports/diagnose/

공통 옵션
    --path 경로          파일 또는 폴더 (기본: data)
    --fs 1000            time 열이 없을 때 센서의 실제 진동 샘플링 [Hz] (time 열도 --fs도 없으면 오류)
    --period-s 60        판정 주기 [s] (기본: config.PERIOD_S). 학습 때 정한 값은 모델에 저장되어 diagnose가 그대로 씀
    --baseline auto|per_unit|reference    정상 기준 방식 (README.md 참고)
    train --no-eval      교차검증 없이 최종 모델만 학습
"""
import argparse
import json
import re
import sys

import config as C
from pipeline import dataio as D, report, workflow as W
from pipeline.heartbeat import Heartbeat
from pipeline.system import AMRPipeline


def show_data(data):
    """읽은 데이터의 로봇별 표와 '확인할 점'(경고)을 화면에 출력"""
    print(data.summary().to_string(index=False))
    if data.warnings:
        print("\n확인할 점:")
        for w in data.warnings:
            print(" -", w)


def show_encoders(data):
    """엔코더 4개 중 몇 개가 살아 있고 얼마나 쓰였는지 (엔코더 열이 있는 로봇만)"""
    rep = D.encoder_report(data)
    if rep.empty:
        return
    print("\n엔코더 상태:")
    print(rep.to_string(index=False))
    for robot, st in data.enc_state.items():
        alive = [k for k, s in st.items() if s["state"] in ("정상", "도중에 멈춤")]
        stalled = [k for k, s in st.items() if s["state"] == "도중에 멈춤"]
        dead = [k for k, s in st.items() if s["state"] in ("죽음", "움직임 없음")]
        msg = f"  {robot}: 엔코더 4개 중 {len(alive)}개 동작" + (f", {len(dead)}개 이상({', '.join(dead)})" if dead else "")
        msg += f" (그중 {', '.join(stalled)}는 도중에 멈춤 → 배선 확인, 오경보 주의)" if stalled else ""
        msg += " → 바퀴 특징 사용" if len(alive) == 4 else " → 바퀴 특징은 쓰지 않고 진동(가속도·각속도)만으로 진단"
        print(msg)


def show_baseline(data, mode):
    """정상 기준 방식과, auto가 공통 기준(reference)을 고른 경우 그 원인이 된 로봇을 출력"""
    chosen = D.choose_baseline(data, mode)
    print(f"\n정상 기준 방식: {chosen} · 판정 주기 {C.PERIOD_S:g}초 · 주기 점수 {quantile_ko(C.PERIOD_QUANTILE)}"
          f" · 상태 변경 {C.CONFIRM_PERIODS}번 연속")
    if chosen == "reference" and (mode or C.BASELINE_MODE) == "auto":
        print(f"  공통 기준을 고른 이유 (자기 처음 {C.minutes(C.BASELINE_PERIODS):g}분으로 기준을 만들 수 없는 로봇): "
              + ", ".join(D.reference_reasons(data)))
        print("  실수로 짧게 끊긴 파일이면 빼고 다시 실행하세요. 그러면 로봇마다 자기 기준(per_unit)을 씁니다.")


def quantile_ko(q):
    """주기 점수 지점 → 화면 문구 (0.5 → '중앙값', 0.75 → '상위 25% 지점')"""
    return "중앙값" if q == 0.5 else f"상위 {round((1 - q) * 100):g}% 지점"


def safe(name):
    """로봇 이름에서 파일 이름에 쓸 수 없는 글자(\\ / : * ? " < > |)를 _로 바꿈 (결과 파일 이름용)"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def at_min(period):
    """판정 주기 번호 → '몇 분째' 문구 (그 주기가 끝나는 시각)"""
    return f"{C.minutes(period + 1):g}분"


def cmd_check(a, hb=None):
    """파일을 읽기만 한다. 로봇별 기록 길이·창 수·샘플링·라벨 표와, 빠진 열·가정한 값 같은 경고를 보여 준다."""
    data = D.load(a.path, a.fs)
    show_data(data)
    show_encoders(data)
    show_baseline(data, a.baseline)


def cmd_train(a, hb=None):
    """두 단계. 1) 성능 측정용 교차검증 (로봇마다 '자신이 빠진 모델'로 평가)
               2) 실제로 쓸 최종 모델은 모든 로봇으로 다시 학습해 저장
    1)의 수치가 발표용 성능이고, 2)의 모델이 diagnose와 대시보드에서 쓰인다."""
    data = D.load(a.path, a.fs)
    show_data(data)
    show_encoders(data)
    show_baseline(data, a.baseline)
    C.REPORT_DIR.mkdir(exist_ok=True)
    metrics = None
    if not a.no_eval and len(data.robots) >= 2:           # 로봇이 1대면 빼고 평가할 수 없으므로 건너뜀
        print("\n[1/2] 로봇 단위 교차검증 (평가 로봇은 학습에 쓰지 않음)", flush=True)
        hb.set("평가 중 (로봇 단위 교차검증: 겹마다 평가 로봇을 빼고 학습 → 그 로봇 진단)")
        metrics = W.cross_validate(data, a.baseline, progress=lambda s: print(s, flush=True))
    print("\n[2/2] 최종 모델 학습 (모든 로봇)", flush=True)
    hb.set("학습 중 (최종 모델: 모든 로봇)")
    pipe, notes = W.train(data, baseline=a.baseline)
    print("저장:", pipe.save())                            # artifacts/model_bundle.joblib 한 파일에 모든 모델이 들어감
    for n in notes:
        print(" -", n)
    if metrics:                                            # 평가 결과 → reports/ 에 수치(json)·그래프(png)·요약(md)
        (C.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        report.fig_auc(metrics, C.REPORT_DIR / "fig_auc.png")
        report.fig_units(metrics, C.REPORT_DIR / "fig_units.png")
        report.fig_confusion(metrics, C.REPORT_DIR / "fig_confusion.png")
        if pipe.bundle["cause"] is not None:
            report.fig_importance(pipe.bundle["cause"].importance(), C.REPORT_DIR / "fig_importance.png")
        report.summary_md(metrics, notes, C.REPORT_DIR / "summary.md")
        # 최종 결과: 판정 정답률·고장 감지율·오경보율 등 %와 로봇별 판정표 (같은 내용이 summary.md 맨 위에도 들어감)
        print("\n".join(report.final_text(metrics)))
        if metrics["skipped"]:                             # 평가하지 못한 로봇과 이유 (빠진 로봇은 위 수치에 들어가지 않음)
            print("평가에서 빠진 항목:")
            for s in metrics["skipped"]:
                print(" -", s)
        print("보고서 (그래프·summary.md·metrics.json):", C.REPORT_DIR)


def cmd_diagnose(a, hb=None):
    """저장된 최종 모델로 로봇마다 처음부터 끝까지 1초 창을 넣어 보고, 판정 주기별 결과 표(CSV)와 점수 그래프(PNG)를 남긴다.
    학습에 쓰인 로봇은 표시해 둔다 (그 로봇의 결과는 성능이 아님)."""
    if not C.BUNDLE_PATH.exists():
        raise D.DataError("모델이 없습니다. 먼저 python run.py train 을 실행하세요.")
    pipe = AMRPipeline.load()
    C.PERIOD_S = pipe.bundle.get("period_s", C.PERIOD_S)    # 학습 때와 같은 판정 주기로 읽어야 기준이 맞음
    # 주기 점수 지점·연속 횟수도 모델에 저장된 학습 때 값을 씀 (infer.Predictor가 직접 읽음. 예전 모델은 중앙값·2번)
    print(f"모델 설정: 판정 주기 {C.PERIOD_S:g}초 · 주기 점수 {quantile_ko(pipe.bundle.get('period_quantile', 0.5))}"
          f" · 상태 변경 {pipe.bundle.get('confirm_periods', 2)}번 연속")
    data = D.load(a.path, a.fs)
    out = C.REPORT_DIR / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    trained = pipe.bundle.get("trained_units", [])
    hb.set("평가 중 (로봇별 진단)")
    for name, recs in data.robots.items():
        try:
            table, mon = W.diagnose(pipe, recs)
        except D.DataError as e:                           # 기록이 너무 짧은 로봇 등은 건너뛰고 다음 로봇으로
            print(f"{name}: 건너뜀 ({e})")
            continue
        table.to_csv(out / f"{safe(name)}.csv", index=False, encoding="utf-8-sig")   # utf-8-sig: 엑셀에서 열어도 한글이 안 깨짐
        report.fig_robot(table, name, "학습에 쓰인 로봇" if name in trained else "처음 보는 로봇", out / f"{safe(name)}.png", mon.calibrated)
        events = [f"{at_min(p)} {C.STATUS_KO[s]}" for p, s in mon.events] or ["없음"]   # 예: '15분 주의', '20분 경고'
        causes = table.loc[table["추정 원인"] != "", "추정 원인"]   # 원인이 적힌 주기들 (마지막 것을 요약에 씀)
        print(f"{name}: 최고 점수 {table['건강 점수'].max():.1f}, 최종 상태 {C.STATUS_KO[mon.status]}, 상태 변경 {', '.join(events)}"
              + (f", 추정 원인 {causes.iloc[-1]}" if len(causes) else "") + (" (학습에 쓰인 로봇)" if name in trained else ""))
        pw = mon.power                                     # 전류·배터리 감시 요약 (current·voltage 열이 있을 때만)
        if pw.has_data:
            vs = [p["voltage"] for p in pw.periods.values() if p["voltage"] is not None]
            print("    전류·배터리: " + ", ".join(
                (["상태 변경 " + (", ".join(f"{at_min(p)} {e}" for p, e in pw.events) or "없음")])
                + ([f"사용량 {pw.used_mah:.0f} mAh (용량의 {pw.used_mah / C.BATTERY_CAPACITY_MAH * 100:.0f}%)"] if pw.has_current else [])
                + ([f"배터리 전압 {vs[0]:.2f} → {vs[-1]:.2f} V"] if vs else [])))
    print("저장 위치:", out)


def main():
    p = argparse.ArgumentParser(description="AMR 고장예지 (실측 데이터 전용)")
    sub = p.add_subparsers(dest="cmd", required=True)    # 첫 번째 단어가 명령 (check / train / diagnose)
    for name in ("check", "train", "diagnose"):           # 세 명령이 공통으로 받는 옵션
        s = sub.add_parser(name)
        s.add_argument("--path", default=str(C.DATA_DIR))
        s.add_argument("--fs", type=float)
        if name != "diagnose":                             # diagnose는 학습 때 정한 판정 주기를 모델에서 읽으므로 받지 않음
            s.add_argument("--period-s", dest="period_s", type=float)
        s.add_argument("--baseline", choices=["auto", "per_unit", "reference"], default=None)
        if name == "train":
            s.add_argument("--no-eval", action="store_true")
    a = p.parse_args()
    if getattr(a, "period_s", None):
        C.PERIOD_S = a.period_s                            # 이번 실행에서만 판정 주기를 바꿈
    try:
        # 지금 하는 작업을 시작할 때 한 번, 그 뒤 5분마다 화면에 알림 (config.HEARTBEAT_S)
        start = {"check": "데이터 검사 중", "train": "데이터 읽는 중", "diagnose": "모델·데이터 읽는 중"}[a.cmd]
        with Heartbeat(start) as hb:
            globals()[f"cmd_{a.cmd}"](a, hb)                # 명령 이름 그대로 cmd_check / cmd_train ... 함수를 부름
    except D.DataError as e:                               # 데이터 문제는 긴 오류 대신 고칠 방법만 보여 줌
        print("오류:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
