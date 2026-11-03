"""시뮬레이터 본체: 경로를 반복 주행하며 1초 창마다 센서 데이터를 만든다

출력: 창(window)마다 한 줄씩
  - 상황 정보 (반복 번호, 구간, 속도, 실제 고장 정도 = 정답 라벨)
  - 좌·우 바퀴의 진동/전류/온도 원시 신호 (raw)
"""
import numpy as np

import config as C
from models import FaultState, MotorModel, fault_severity, vibration_signal, road_noise
from sensors import accelerometer, current_sensor, temperature_sensor

WHEELS = ("left", "right")


class AMRSimulator:
    def __init__(self, route=C.ROUTE, faults=C.FAULTS, seed=C.SEED):
        self.route = route
        self.faults = faults
        self.rng = np.random.default_rng(seed)
        self.motors = {w: MotorModel(C.MOTOR, C.ROBOT) for w in WHEELS}
        self.fault_state = {w: FaultState() for w in WHEELS}

    # --------------------------------------------------------
    def _update_faults(self, lap):
        for w in WHEELS:
            self.fault_state[w] = FaultState()
        for f in self.faults:
            setattr(self.fault_state[f["wheel"]], f["kind"], fault_severity(f, lap))

    def _segment_windows(self, seg):
        """구간을 1초 창들로 나누고, 문턱 위치를 창에 배치"""
        duration = seg["length"] / seg["speed"]
        n_win = max(1, int(round(duration / C.WINDOW_S)))
        bump_wins = set(self.rng.choice(n_win, size=min(seg["bumps"], n_win), replace=False))
        return n_win, bump_wins

    # --------------------------------------------------------
    def run(self):
        """전체 시뮬레이션 실행 → 창 단위 기록 리스트 반환"""
        records = []
        fs = C.PHYS_FS
        t = np.arange(int(fs * C.WINDOW_S)) / fs
        prev_speed = 0.0

        for lap in range(C.N_LAPS):
            self._update_faults(lap)
            for seg in self.route:
                n_win, bump_wins = self._segment_windows(seg)
                for k in range(n_win):
                    speed = seg["speed"]
                    accel = (speed - prev_speed) / C.WINDOW_S   # 구간 진입 시 가감속
                    prev_speed = speed
                    omega = speed / C.ROBOT["wheel_radius"]
                    bumps = [self.rng.uniform(0.1, 0.9)] if k in bump_wins else []
                    shared_road = road_noise(len(t), self.rng, fs)   # 좌우 공통 노면

                    rec = dict(lap=lap, segment=seg["name"], speed=speed,
                               accel=accel, roughness=seg["roughness"], bump=len(bumps) > 0)
                    for w in WHEELS:
                        fault = self.fault_state[w]
                        # 진동
                        phys = vibration_signal(t, speed, C.ROBOT["wheel_radius"], seg["roughness"],
                                                bumps, fault, shared_road, self.rng)
                        vib = accelerometer(phys, fs, C.SENSORS, self.rng)
                        # 전류
                        n_cur = int(C.SENSORS["cur_fs"] * C.WINDOW_S)
                        i_true = self.motors[w].current(accel, fault)
                        cur = current_sensor(np.full(n_cur, i_true), C.SENSORS, self.rng)
                        # 온도
                        temp_true = self.motors[w].step_temperature(i_true, omega, fault, C.WINDOW_S)
                        temp = temperature_sensor(temp_true, C.SENSORS)

                        rec[f"{w}_vib"] = vib
                        rec[f"{w}_cur"] = cur
                        rec[f"{w}_temp"] = float(temp)
                        # 정답 라벨 (평가용)
                        for kind, val in fault.as_dict().items():
                            rec[f"{w}_true_{kind}"] = val
                    records.append(rec)
        return records
