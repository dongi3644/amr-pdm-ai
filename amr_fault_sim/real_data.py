"""실측 데이터 연결 (나중에 실물 로봇 데이터로 교체할 때 사용)

아두이노/라즈베리파이에서 아래 형식의 CSV를 기록하면
시뮬레이터 대신 이 함수의 결과를 detector.extract()에 그대로 넣을 수 있다.

CSV 형식 (1행 = 1샘플, 진동 센서 주기로 기록):
    time_s, lap, segment, speed, left_ax, right_ax, left_cur, right_cur, left_temp, right_temp, label
    - label: 결함 조건 (예: normal, unbalance_1, loose_2 ...) → 정답 라벨
"""
import csv
from collections import defaultdict

import numpy as np

import config as C


def load_csv(path):
    """실측 CSV → 시뮬레이터와 같은 형식의 창(window) 기록 리스트"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = int(C.SENSORS["vib_fs"] * C.WINDOW_S)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lap"], r["segment"], r["label"])].append(r)

    records = []
    for (lap, seg, label), g in groups.items():
        for i in range(0, len(g) - n + 1, n):           # 1초씩 자르기
            w = g[i:i + n]
            rec = dict(lap=int(lap), segment=seg, speed=float(w[0]["speed"]),
                       accel=0.0, roughness=0.0, bump=False)
            sev = 0.0 if label == "normal" else 1.0
            for side in ("left", "right"):
                rec[f"{side}_vib"] = np.array([float(x[f"{side}_ax"]) for x in w])
                rec[f"{side}_cur"] = np.array([float(x[f"{side}_cur"]) for x in w])
                rec[f"{side}_temp"] = float(np.mean([float(x[f"{side}_temp"]) for x in w]))
                for kind in ("unbalance", "looseness", "bearing"):
                    rec[f"{side}_true_{kind}"] = sev if kind in label else 0.0
            records.append(rec)
    return records
