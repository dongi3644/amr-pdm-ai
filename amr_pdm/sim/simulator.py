"""AMR 주행 시뮬레이터: 경로를 반복 주행하며 1초 창마다 좌·우 바퀴 센서 데이터를 만든다.

실물 로봇이 생기면 이 파일 대신 실측 데이터를 같은 형식으로 넣으면 된다.
"""
import numpy as np

import config as C
from sim.physics import FaultState, MotorModel, vibration_signal, road_noise
from sim.sensors import accelerometer, current_sensor, temperature_sensor

WHEELS = ("left", "right")


def severity(fault, lap):
    """고장 시나리오: start_lap부터 end_lap까지 0 → max 로 선형 증가"""
    s, e, m = fault["start_lap"], fault["end_lap"], fault.get("max_severity", 1.0)
    if lap < s:
        return 0.0
    return float(min(1.0, (lap - s) / max(1, e - s)) * m)


class AMRSimulator:
    """faults: [{kind, wheel, start_lap, end_lap, max_severity}]
    variation: 로봇마다 다른 질량·마찰·주변온도 + 회차마다 달라지는 적재·바닥·온도"""

    def __init__(self, faults=(), n_laps=40, seed=C.SEED, variation=True, route=C.ROUTE):
        self.faults, self.n_laps, self.route, self.variation = list(faults), n_laps, route, variation
        self.rng = np.random.default_rng(seed)
        robot, motor = dict(C.ROBOT), dict(C.MOTOR)
        if variation:
            robot["mass"] *= self.rng.uniform(0.85, 1.2)          # 적재물 차이
            robot["rolling_coef"] *= self.rng.uniform(0.9, 1.1)
            motor["ambient"] = self.rng.uniform(20, 28)
        self.robot, self.motor = robot, motor
        self.motors = {w: MotorModel(motor, robot) for w in WHEELS}
        for m in self.motors.values():                             # 운행 전 예열 → 온도 정상상태
            for _ in range(3000):
                m.step_temperature(m.current(0.0, FaultState()), 1.0 / robot["wheel_radius"], FaultState(), 1.0)

    def repair(self):
        """정비 완료: 다음 회차부터 고장 성분이 사라진다 (모터 온도는 서서히 내려감)"""
        self.faults = []

    def _fault_state(self, lap):
        st = {w: FaultState() for w in WHEELS}
        for f in self.faults:
            setattr(st[f["wheel"]], f["kind"], severity(f, lap))
        return st

    def windows(self):
        """1초 창 단위 기록을 하나씩 생성 (실시간 스트리밍처럼 사용 가능)"""
        fs, rng = C.PHYS_FS, self.rng
        t = np.arange(int(fs * C.WINDOW_S)) / fs
        prev_speed = 0.0
        base_mass, base_ambient = self.robot["mass"], self.motor["ambient"]
        amb_phase = rng.uniform(0, 2 * np.pi)
        for lap in range(self.n_laps):
            fstate = self._fault_state(lap)
            rough_k = 1.0
            if self.variation:                                     # MotorModel이 같은 dict를 참조하므로 바로 반영됨
                self.robot["mass"] = base_mass * rng.uniform(*C.LAP_LOAD)
                rough_k = rng.uniform(*C.LAP_ROUGHNESS)
                self.motor["ambient"] = base_ambient + C.AMBIENT_SWING * np.sin(
                    2 * np.pi * lap / C.AMBIENT_PERIOD_LAPS + amb_phase)
            for seg in self.route:
                n_win = max(1, int(round(seg["length"] / seg["speed"] / C.WINDOW_S)))
                bump_wins = set(rng.choice(n_win, size=min(seg["bumps"], n_win), replace=False).tolist())
                for k in range(n_win):
                    speed = seg["speed"] * (1 + rng.normal(0, 0.02))
                    accel = (speed - prev_speed) / C.WINDOW_S
                    prev_speed = speed
                    omega = speed / self.robot["wheel_radius"]
                    bumps = [rng.uniform(0.1, 0.9)] if k in bump_wins else []
                    shared = road_noise(len(t), rng, fs)
                    rec = dict(lap=lap, segment=seg["name"], seg_pos=k, speed=speed, accel=accel,
                               bump=bool(bumps))
                    for w in WHEELS:
                        fault = fstate[w]
                        phys = vibration_signal(t, speed, self.robot["wheel_radius"], seg["roughness"] * rough_k,
                                                bumps, fault, shared, rng)
                        rec[f"{w}_vib"] = accelerometer(phys, fs, C.SENSORS, rng)
                        i_true = self.motors[w].current(accel, fault)
                        rec[f"{w}_cur"] = current_sensor(np.full(int(C.SENSORS["cur_fs"] * C.WINDOW_S), i_true),
                                                         C.SENSORS, rng)
                        temp = self.motors[w].step_temperature(i_true, omega, fault, C.WINDOW_S)
                        rec[f"{w}_temp"] = float(temperature_sensor(temp, C.SENSORS))
                        for kind, v in fault.as_dict().items():
                            rec[f"{w}_true_{kind}"] = v          # 정답 라벨 (평가용, 실제 운영에서는 없음)
                    yield rec

    def run(self):
        return list(self.windows())
