"""물리 모델: 모터(전류·온도), 진동, 고장 주입

실제 물리를 정밀하게 재현하는 것이 아니라,
'고장이 생기면 어떤 신호가 어떻게 변하는지'를 단순한 식으로 흉내 내는 것이 목적.
계수는 실측 데이터를 얻으면 그에 맞춰 조정(보정)한다.
"""
import numpy as np

G = 9.81


# ============================================================
# 1. 고장 상태
# ============================================================
class FaultState:
    """바퀴 하나의 고장 정도 (0 = 정상, 1 = 심각)"""

    def __init__(self):
        self.unbalance = 0.0   # 불평형: 회전 주파수(1X) 진동 증가
        self.looseness = 0.0   # 헐거움: 고조파 + 충격성 진동
        self.bearing = 0.0     # 베어링: 결함 주파수 충격 + 마찰(전류·온도) 증가

    def friction_factor(self):
        """고장으로 늘어나는 추가 마찰 (전류·온도에 반영)"""
        return 1.0 + 0.6 * self.bearing + 0.15 * self.looseness + 0.1 * self.unbalance

    def as_dict(self):
        return dict(unbalance=self.unbalance, looseness=self.looseness, bearing=self.bearing)


def fault_severity(fault_cfg, lap):
    """고장 시나리오에 따라 현재 반복(lap)에서의 고장 정도 계산"""
    s, e, m = fault_cfg["start_lap"], fault_cfg["end_lap"], fault_cfg["max_severity"]
    if lap < s:
        return 0.0
    return float(min(1.0, (lap - s) / max(1, e - s)) * m)


# ============================================================
# 2. 모터 모델 (전류 + 온도)
# ============================================================
class MotorModel:
    def __init__(self, motor_cfg, robot_cfg):
        self.m = motor_cfg
        self.r = robot_cfg
        self.temp = motor_cfg["ambient"]

    def current(self, accel, fault: FaultState):
        """바퀴 하나가 내야 하는 토크 → 전류 [A]"""
        mass_per_wheel = self.r["mass"] / 2
        f_roll = self.r["rolling_coef"] * mass_per_wheel * G * fault.friction_factor()
        f_acc = mass_per_wheel * accel
        torque = (f_roll + f_acc) * self.r["wheel_radius"]
        return self.m["i_noload"] * fault.friction_factor() + abs(torque) / self.m["kt"]

    def step_temperature(self, current, omega, fault: FaultState, dt):
        """1차 열 모델: 저항 발열 + 마찰 발열 - 방열"""
        p_joule = current ** 2 * self.m["resistance"]
        p_fric = 0.08 * abs(omega) * (fault.friction_factor() - 1.0)
        p_out = (self.temp - self.m["ambient"]) / self.m["thermal_res"]
        self.temp += (p_joule + p_fric - p_out) / self.m["thermal_cap"] * dt
        return self.temp


# ============================================================
# 3. 진동 모델
# ============================================================
def vibration_signal(t, speed, wheel_radius, roughness, bump_times,
                     fault: FaultState, road_noise, rng):
    """한 바퀴 위치의 가속도 신호 [g]를 고속 샘플(t)로 생성

    road_noise: 좌·우 바퀴가 공유하는 노면 진동 (같은 바닥을 지나므로)
    """
    omega = speed / wheel_radius              # 바퀴 각속도 [rad/s]
    fr = omega / (2 * np.pi)                  # 회전 주파수 [Hz]
    phase = 2 * np.pi * fr * t

    # 기본 구동 진동 (모터·기어 맞물림 등)
    sig = 0.01 * np.sin(phase) + 0.005 * np.sin(2 * phase + 0.3)

    # 노면 (좌우 공통 성분 + 바퀴 개별 성분)
    sig += roughness * (0.6 * road_noise + 0.15 * rng.standard_normal(len(t)) * 0.1)

    # 문턱 충격: 감쇠 진동
    for tb in bump_times:
        dt = t - tb
        mask = dt >= 0
        sig[mask] += 0.8 * np.exp(-dt[mask] * 40) * np.sin(2 * np.pi * 35 * dt[mask])

    # --- 고장 성분 ---
    if fault.unbalance > 0:      # 1X 성분, 속도 제곱에 비례
        sig += fault.unbalance * 0.05 * (omega / 10) ** 2 * np.sin(phase + 0.7)

    if fault.looseness > 0:      # 1X~5X 고조파 + 회전당 1회 충격
        for k in range(1, 6):
            sig += fault.looseness * 0.02 / k * np.sin(k * phase + k)
        impulses = (np.mod(phase, 2 * np.pi) < 0.15).astype(float)
        sig += fault.looseness * 0.15 * impulses

    if fault.bearing > 0:        # 외륜 결함 주파수(BPFO≈3.6×회전수) 충격이 고주파 공진을 때림
        bpfo = 3.6 * fr
        hits = np.mod(t * bpfo, 1.0) < 0.02
        resonance = np.sin(2 * np.pi * 1800 * t)          # 고주파 공진 (센서 대역 밖일 수 있음)
        sig += fault.bearing * 0.3 * hits * resonance
        sig += fault.bearing * 0.03 * np.sin(2 * np.pi * bpfo * t)  # 저주파 쪽으로 새는 성분

    return sig


def road_noise(n, rng, fs):
    """좌우 바퀴가 공유하는 노면 진동 (저역 위주 무작위 신호)"""
    white = rng.standard_normal(n)
    k = max(1, int(fs / 200))                 # 간단한 이동평균 저역통과
    return np.convolve(white, np.ones(k) / k, mode="same") * 0.3
