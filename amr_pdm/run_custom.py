"""내 데이터로 실행: CSV를 user_data 폴더에 넣고 아래 순서로 실행한다. 형식 설명은 DATA_FORMAT.md

    python run_custom.py check                    # 1) 파일 형식 검사
    python run_custom.py train                    # 2) (선택) 내 데이터로 학습 → artifacts/user_model.joblib
    python run_custom.py diagnose                 # 3) 진단 → reports/user/ 에 회차별 CSV와 그래프 저장

    python run_custom.py sample                   # 파일이 없을 때: 시뮬레이터로 예시 파일 생성

공통 옵션
    --path 경로          파일 또는 폴더 (기본: user_data)
    --fs 1000            time 열이 없을 때 진동 샘플링 [Hz]
    --lap-s 60           lap 열이 없을 때 1회차 길이 [s]
    train    --holdout R3,R4     학습에서 뺀 뒤 성능 확인에 쓸 로봇
    diagnose --model auto|user|sim   진단 모델 (auto: 내 모델이 있으면 내 모델)
"""
import argparse
import re
import sys

import config as C
from pipeline import report, userdata as U
from pipeline.system import BUNDLE_PATH, AMRPipeline


def pick_model(choice):
    user, sim = C.USER_BUNDLE_PATH.exists(), BUNDLE_PATH.exists()
    if choice == "user" or (choice == "auto" and user):
        if not user:
            raise U.UserDataError("내 데이터로 학습한 모델이 없습니다. 먼저 python run_custom.py train 을 실행하세요.")
        return AMRPipeline.load(C.USER_BUNDLE_PATH), "내 데이터로 학습한 모델"
    if not sim:
        raise U.UserDataError("모델이 없습니다. python run_custom.py train 또는 python run_all.py 를 먼저 실행하세요.")
    return AMRPipeline.load(BUNDLE_PATH), "시뮬레이션으로 학습한 모델"


def show_data(data):
    print(data.summary().to_string(index=False))
    if data.warnings:
        print("\n확인할 점:")
        for w in data.warnings:
            print(" -", w)


def save_result(name, table, model_name, mon):
    C.USER_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    table.to_csv(C.USER_REPORT_DIR / f"{safe}.csv", index=False, encoding="utf-8-sig")
    report.fig_user(table, name, model_name, C.USER_REPORT_DIR / f"{safe}.png", mon.calibrated)


def print_result(name, table, mon):
    worst = table["건강 점수"].max()
    causes = table.loc[table["추정 원인"] != "", "추정 원인"]
    events = [(lap, C.STATUS_KO[s]) for lap, s in mon.events] or "없음"
    print(f"{name}: 최고 점수 {worst:.1f}, 최종 상태 {C.STATUS_KO[mon.status]}, 상태 변경 {events}"
          + (f", 마지막 추정 원인 {causes.iloc[-1]}" if len(causes) else ""))


def cmd_check(a):
    show_data(U.load(a.path, a.fs, a.lap_s))


def cmd_sample(a):
    C.USER_DATA_DIR.mkdir(exist_ok=True)
    plans = [("normal_A", [], 701), ("normal_B", [], 702),
             ("bearing_right", [dict(kind="bearing", wheel="right", start_lap=8, end_lap=26)], 703)]
    for name, faults, seed in plans:
        path = C.USER_DATA_DIR / f"{U.SAMPLE_PREFIX}{name}.csv.gz"
        print("생성:", U.write_sample(path, faults, n_laps=a.laps, seed=seed), flush=True)


def cmd_train(a):
    data = U.load(a.path, a.fs, a.lap_s)
    show_data(data)
    holdout = [h.strip() for h in a.holdout.split(",") if h.strip()]
    unknown = [h for h in holdout if h not in data.robots]
    if unknown:
        raise U.UserDataError(f"--holdout에 없는 로봇: {unknown}. 로봇 목록: {list(data.robots)}")
    print("\n학습 중...", flush=True)
    pipe, notes = U.train_user(data, holdout)
    print("저장:", pipe.save(C.USER_BUNDLE_PATH))
    for n in notes:
        print(" -", n)
    for name in holdout:
        table, mon = U.diagnose(pipe, data.robots[name])
        save_result(name, table, "학습에서 뺀 로봇", mon)
        print_result(name, table, mon)


def cmd_diagnose(a):
    data = U.load(a.path, a.fs, a.lap_s)
    show_data(data)
    pipe, model_name = pick_model(a.model)
    print(f"\n모델: {model_name}")
    if pipe.bundle["source"] != "user":
        print("  시뮬레이션 눈금이라 점수와 원인은 참고용입니다. 실제 판단에는 내 데이터로 학습한 모델을 쓰세요.")
    trained = pipe.bundle.get("trained_robots", [])
    for name, recs in data.robots.items():
        try:
            table, mon = U.diagnose(pipe, recs)
        except U.UserDataError as e:
            print(f"{name}: 건너뜀 ({e})")
            continue
        save_result(name, table, model_name, mon)
        print_result(name, table, mon)
        if name in trained:
            print("  이 로봇은 학습에 쓰였으므로 실제 성능으로 보면 안 됩니다.")
    print("저장 위치:", C.USER_REPORT_DIR)


def main():
    p = argparse.ArgumentParser(description="내 데이터로 AMR 고장예지 실행")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "train", "diagnose"):
        s = sub.add_parser(name)
        s.add_argument("--path", default=str(C.USER_DATA_DIR))
        s.add_argument("--fs", type=float)
        s.add_argument("--lap-s", dest="lap_s", type=float)
        if name == "train":
            s.add_argument("--holdout", default="")
        if name == "diagnose":
            s.add_argument("--model", choices=["auto", "user", "sim"], default="auto")
    sub.add_parser("sample").add_argument("--laps", type=int, default=30)
    a = p.parse_args()
    try:
        globals()[f"cmd_{a.cmd}"](a)
    except U.UserDataError as e:
        print("오류:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
