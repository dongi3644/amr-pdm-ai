# -*- coding: utf-8 -*-
"""외부 회전기계 데이터(충남대 LNG 재액화 테스트베드) → 우리 모델 입력 형식(로봇 CSV)으로 늘려 맞추기

목적: 우리 프로젝트(배달 로봇)와 맞는 데이터는 아니지만, 실제로 잰 진동으로 모델이 끝까지 잘 돌아가는지 확인한다.
      여기서 나온 숫자는 우리 로봇의 성능이 아니다 (발표 성능으로 쓰지 말 것).

[원본] 메인 축 2,300 rpm, 6,000 Hz, 기록 하나 약 59초, 센서 2개 (m = 메인 축 베어링, p = 피니언 축 베어링, 같은 시각에 기록)
       정상 10개 · 메인 축 불평형 3단계 × 6개 · 피니언 축 불평형 2단계 × 6개 · 기어 이 빠짐 3단계 × 6개
       파일 이름: Gearfault1_2_m.csv = 기어 결함 1단계, 2번째 기록, 메인 축 센서. 열: 시간, 진동 (머리줄 없음)

[맞추는 방법]
 1) 로봇 1대 = 정상 기록 6개(약 6분, 처음 5분이 그 로봇의 '평소' 기준이 됨) 다음에 고장 기록을 1단계 → 2단계 → 3단계 순서로 이어 붙임
    (고장이 점점 심해지는 로봇처럼 보이게). 정상 로봇은 정상 기록 8개를 이어 붙임
 2) 정상 기록이 10개뿐이라 여러 로봇이 같은 정상 기록을 다시 씀. 그래서 로봇마다 다음 두 가지를 다르게 해서 '개체 차이'를 줌
    - 시간 늘이기(스트레칭): 로봇마다 신호를 0.97~1.03배 빠르기로 다시 샘플링 (회전 속도가 조금씩 다른 로봇처럼)
    - 크기: 로봇마다 0.9~1.1배 (센서를 붙인 상태가 조금씩 다른 로봇처럼)
    한 로봇 안에서는 정상·고장 구간 모두 같은 배율을 씀 (기준과 비교가 공정하도록)
 3) 우리 모델은 MPU6050 6축(ax, ay, az, gx, gy, gz)이 모두 있어야 하는데 원본은 센서 2개 × 1축뿐이라, 6개 열을 실제 측정값으로만 채움
      ax = 메인 축 센서                     ay = 피니언 축 센서                    az = 1 g(중력) + (메인 + 피니언) / 2
      gx = 메인 축 고주파(1~2.8 kHz) 포락선   gy = 피니언 축 고주파 포락선           gz = (메인 − 피니언) / 2
    포락선: 6 kHz로 잰 기어 맞물림(약 1.9 kHz) 근처 떨림의 크기 변화. 1 kHz로 줄이면 사라지는 정보를 남기려고 넣음
    az의 1 g는 모델이 '가속도 크기가 1 g 근처인가'를 검사하기 때문에 더한 값 (진동 특징은 평균을 빼고 계산하므로 결과에 영향 없음)
 4) 6,000 Hz → 1,000 Hz로 줄임 (우리 로봇 기록과 비슷한 크기, 파일 크기도 줄임). 시간은 t_ms(정수 밀리초) 열로 새로 만듦
 5) 라벨: fault_kind = 주축 불평형 / 피니언축 불평형 / 기어 결함, fault_severity = 단계별 고장 정도
      3단계 고장: 0.4 → 0.8 → 1.0,  2단계 고장(피니언): 0.4 → 0.8   (원본에는 '정도' 값이 없어 단계 순서로 정한 값)

[실행] python 외부데이터_시험/만들기.py [--src 원본폴더] [--out 출력폴더] [--fs 1000] [--seed 7] [--normal-first 6]
       → 외부데이터_시험/data/ 에 로봇마다 .csv.gz 하나 (압축된 CSV, 모델이 그대로 읽음)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
SRC_FS = 6000
DEFAULT_SRC = Path.home() / "Downloads" / "csv file-20261110T132436Z-1-001" / "csv file"

# 고장 종류: (원본 폴더 앞부분, 파일 이름 앞부분, 단계 수, 라벨 이름)
FAULTS = {
    "MS": ("Main Imbalance", "MainImbalance", 3, "주축 불평형"),
    "PS": ("Pinion Imbalance", "PinionImbalance", 2, "피니언축 불평형"),
    "GF": ("Gearfault", "Gearfault", 3, "기어 결함"),
}
SEVERITY = {3: [0.4, 0.8, 1.0], 2: [0.4, 0.8]}
ROBOTS_PER_FAULT = 3          # 고장 종류마다 로봇 3대 (단계마다 기록 6개 → 로봇마다 2개씩)
NORMAL_ROBOTS = 4             # 정상만 달리는 로봇 수
NORMAL_FIRST = 6              # 고장 로봇의 처음 정상 기록 수 (약 6분 > 기준 5분)
NORMAL_ONLY = 8               # 정상 로봇의 정상 기록 수 (약 8분)


def read(path):
    """원본 CSV(머리줄 없음: 시간, 진동) → 진동 값 배열. 시간 열은 파일마다 단위가 달라(초/밀리초) 쓰지 않고 6,000 Hz로 다시 셈"""
    return pd.read_csv(path, header=None, usecols=[1], dtype=np.float64).iloc[:, 0].to_numpy()


def recording(src, folder, stem):
    """같은 기록의 메인 축(m)·피니언 축(p) 신호를 같은 길이로"""
    m, p = read(src / folder / f"{stem}_m.csv"), read(src / folder / f"{stem}_p.csv")
    n = min(len(m), len(p))
    return m[:n], p[:n]


BAND = signal.butter(4, [1000, 2800], btype="bandpass", fs=SRC_FS, output="sos")   # 포락선을 볼 고주파 대역


def to_robot_format(m, p, stretch, gain, fs_out):
    """6 kHz 두 신호 → fs_out Hz 6축. stretch배 빠르게(다시 샘플링), gain배 크게"""
    up, down = 100, int(round(100 * stretch * SRC_FS / fs_out))    # 6 kHz → (fs_out / stretch) Hz로 뽑아서 fs_out Hz로 읽음 = stretch배 빠름

    def low(x):                                                    # 다시 샘플링 (거르기 포함: 새 나이퀴스트 넘는 떨림이 섞여 들지 않게)
        return signal.resample_poly(x, up, down)

    def env(x):                                                    # 고주파 대역의 크기 변화(포락선) → 다시 샘플링
        return low(np.abs(signal.hilbert(signal.sosfiltfilt(BAND, x))))

    m, p = m * gain, p * gain
    lm, lp = low(m), low(p)
    em, ep = env(m), env(p)
    n = min(len(lm), len(lp), len(em), len(ep))
    lm, lp, em, ep = lm[:n], lp[:n], em[:n], ep[:n]
    return dict(ax=lm, ay=lp, az=1.0 + (lm + lp) / 2, gx=em, gy=ep, gz=(lm - lp) / 2)


def build_plan(rng, normal_first=NORMAL_FIRST):
    """로봇 이름 → [(원본 폴더, 파일 이름 앞부분, 라벨, 고장 정도), ...]
    normal_first: 고장 로봇의 처음 정상 기록 수. 줄여도 같은 기록·같은 배율에서 앞쪽만 쓰도록 무작위 뽑기는 늘 6개로 함"""
    normals = [("Normal", f"Normal{i}") for i in range(1, 11)]
    plan = {}
    for i in range(NORMAL_ROBOTS):
        pick = rng.choice(len(normals), NORMAL_ONLY, replace=False)
        plan[f"정상_{i + 1:02d}"] = [(*normals[j], "normal", 0.0) for j in pick]
    for code, (folder, prefix, levels, kind) in FAULTS.items():
        for r in range(ROBOTS_PER_FAULT):
            pick = rng.choice(len(normals), NORMAL_FIRST, replace=False)
            segs = [(*normals[j], "normal", 0.0) for j in pick[:normal_first]]
            for lv in range(1, levels + 1):
                for k in (2 * r + 1, 2 * r + 2):                  # 로봇 r은 단계마다 기록 2r+1, 2r+2번
                    segs.append((f"{folder}{lv}", f"{prefix}{lv}_{k}", kind, SEVERITY[levels][lv - 1]))
            plan[f"{code}_{r + 1:02d}"] = segs
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="원본 'csv file' 폴더")
    ap.add_argument("--out", type=Path, default=HERE / "data", help="만든 로봇 CSV를 둘 폴더")
    ap.add_argument("--fs", type=int, default=1000, help="만들 샘플링 [Hz] (기본 1000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--normal-first", type=int, default=NORMAL_FIRST,
                    help="고장 로봇의 처음 정상 주행 기록 수 (기록 하나 약 1분, 기본 6). 3·4로 줄이면 기준 시간 비교용")
    a = ap.parse_args()
    if not a.src.is_dir():
        sys.exit(f"원본 폴더가 없습니다: {a.src}  (--src 로 알려 주세요)")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    plan = build_plan(rng, a.normal_first)
    cache = {}
    rows = []
    for robot, segs in plan.items():
        stretch = float(rng.choice([0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]))
        gain = float(np.round(rng.uniform(0.9, 1.1), 3))
        parts = []
        for folder, stem, kind, sev in segs:
            if (folder, stem) not in cache:
                cache[(folder, stem)] = recording(a.src, folder, stem)
            ch = to_robot_format(*cache[(folder, stem)], stretch, gain, a.fs)
            df = pd.DataFrame(ch)
            df["fault_kind"], df["fault_severity"], df["source"] = kind, sev, stem
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df.insert(0, "t_ms", np.arange(len(df)) * 1000 // a.fs)     # 시간: 정수 밀리초 (모델이 초로 바꿔 읽음)
        src_list = df.pop("source")
        out = a.out / f"{robot}.csv.gz"
        df.to_csv(out, index=False, float_format="%.6g", compression="gzip")
        normal_s = float((df["fault_kind"] == "normal").sum()) / a.fs
        rows.append(dict(로봇=robot, 길이_분=round(len(df) / a.fs / 60, 1), 정상_분=round(normal_s / 60, 1),
                         고장=", ".join(dict.fromkeys(k for k in df["fault_kind"] if k != "normal")) or "-",
                         빠르기=stretch, 크기=gain, 쓴_기록=" ".join(dict.fromkeys(src_list))))
        print(f"{robot}: {rows[-1]['길이_분']}분 (정상 {rows[-1]['정상_분']}분), 빠르기 ×{stretch}, 크기 ×{gain} → {out.name}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "만든_로봇_목록.txt", index=False, sep="\t", encoding="utf-8-sig")
    print(f"\n끝: 로봇 {len(rows)}대 → {a.out}")


if __name__ == "__main__":
    main()
