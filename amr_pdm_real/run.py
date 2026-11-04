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
from pipeline.system import AMRPipeline


def show_data(data):
    """읽은 데이터의 로봇별 표와 '확인할 점'(경고)을 화면에 출력"""
    print(data.summary().to_string(index=False))
    if data.warnings:
        print("\n확인할 점:")
        for w in data.warnings:
            print(" -", w)


def safe(name):
    """로봇 이름에서 파일 이름에 쓸 수 없는 글자(\\ / : * ? " < > |)를 _로 바꿈 (결과 파일 이름용)"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def at_min(period):
    """판정 주기 번호 → '몇 분째' 문구 (그 주기가 끝나는 시각)"""
    return f"{C.minutes(period + 1):g}분"


def cmd_check(a):
    """파일을 읽기만 한다. 로봇별 기록 길이·창 수·샘플링·라벨 표와, 빠진 열·가정한 값 같은 경고를 보여 준다."""
    data = D.load(a.path, a.fs)
    show_data(data)
    print(f"\n정상 기준 방식: {D.choose_baseline(data, a.baseline)} · 판정 주기 {C.PERIOD_S:g}초")


def cmd_train(a):
    """두 단계. 1) 성능 측정용 교차검증 (로봇마다 '자신이 빠진 모델'로 평가)
               2) 실제로 쓸 최종 모델은 모든 로봇으로 다시 학습해 저장
    1)의 수치가 발표용 성능이고, 2)의 모델이 diagnose와 대시보드에서 쓰인다."""
    data = D.load(a.path, a.fs)
    show_data(data)
    C.REPORT_DIR.mkdir(exist_ok=True)
    metrics = None
    if not a.no_eval and len(data.robots) >= 2:           # 로봇이 1대면 빼고 평가할 수 없으므로 건너뜀
        print("\n[1/2] 로봇 단위 교차검증 (평가 로봇은 학습에 쓰지 않음)", flush=True)
        metrics = W.cross_validate(data, a.baseline, progress=lambda s: print(s, flush=True))
    print("\n[2/2] 최종 모델 학습 (모든 로봇)", flush=True)
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
        brief = {k: metrics[k] for k in ("protocol", "detectors", "units", "forecast", "health_indicator")}
        brief["cause"] = {k: metrics["cause"][k] for k in ("window_accuracy", "unit_accuracy")}
        print("\n평가 요약:", json.dumps(brief, ensure_ascii=False, indent=1))
        print("보고서:", C.REPORT_DIR)


def cmd_diagnose(a):
    """저장된 최종 모델로 로봇마다 처음부터 끝까지 1초 창을 넣어 보고, 판정 주기별 결과 표(CSV)와 점수 그래프(PNG)를 남긴다.
    학습에 쓰인 로봇은 표시해 둔다 (그 로봇의 결과는 성능이 아님)."""
    if not C.BUNDLE_PATH.exists():
        raise D.DataError("모델이 없습니다. 먼저 python run.py train 을 실행하세요.")
    pipe = AMRPipeline.load()
    C.PERIOD_S = pipe.bundle.get("period_s", C.PERIOD_S)    # 학습 때와 같은 판정 주기로 읽어야 기준이 맞음
    data = D.load(a.path, a.fs)
    out = C.REPORT_DIR / "diagnose"
    out.mkdir(parents=True, exist_ok=True)
    trained = pipe.bundle.get("trained_units", [])
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
        globals()[f"cmd_{a.cmd}"](a)                       # 명령 이름 그대로 cmd_check / cmd_train ... 함수를 부름
    except D.DataError as e:                               # 데이터 문제는 긴 오류 대신 고칠 방법만 보여 줌
        print("오류:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
