"""시뮬레이션 설정값 (필요한 값만 여기서 바꾸면 됨)"""

# ---------------- 로봇 ----------------
ROBOT = dict(
    mass=20.0,            # 로봇 + 적재물 질량 [kg]
    wheel_radius=0.08,    # 구동바퀴 반지름 [m]
    rolling_coef=0.015,   # 구름저항 계수
)

# ---------------- 모터 (소형 BLDC/DC 모터 근사) ----------------
MOTOR = dict(
    kt=0.35,              # 토크 상수 [Nm/A]
    i_noload=0.25,        # 무부하 전류 [A]
    resistance=2.0,       # 권선 저항 [ohm]
    thermal_cap=60.0,     # 열용량 [J/K]
    thermal_res=8.0,      # 열저항 (모터→공기) [K/W]
    ambient=24.0,         # 주변 온도 [°C]
)

# ---------------- 센서 ----------------
SENSORS = dict(
    vib_fs=1000,          # 진동 센서 샘플링 [Hz] (MPU6050 최대치 근사)
    vib_range_g=4.0,      # 측정 범위 [g]
    vib_noise_g=0.004,    # 잡음 표준편차 [g]
    cur_fs=100,           # 전류 센서 샘플링 [Hz]
    cur_noise_a=0.03,     # 전류 잡음 [A]
    temp_fs=1,            # 온도 센서 샘플링 [Hz]
    temp_res=0.0625,      # 온도 분해능 [°C] (DS18B20 근사)
)

# 실제 진동은 센서보다 훨씬 빠르게 일어나므로 내부적으로 고속 생성 후 센서가 걸러냄
PHYS_FS = 5000            # 내부 물리 신호 생성 주파수 [Hz]
WINDOW_S = 1.0            # 분석 창 길이 [s]

# ---------------- 경로 (구간별 노면) ----------------
# length[m], speed[m/s], roughness(노면 거칠기 0~1), bumps(문턱 개수)
ROUTE = [
    dict(name="로비",     length=12, speed=0.8, roughness=0.10, bumps=0),
    dict(name="복도A",    length=20, speed=1.0, roughness=0.15, bumps=1),
    dict(name="엘리베이터앞", length=6, speed=0.5, roughness=0.30, bumps=2),
    dict(name="복도B",    length=18, speed=1.0, roughness=0.20, bumps=0),
]

# ---------------- 고장 시나리오 ----------------
# start_lap부터 severity가 0 → max_severity까지 선형 증가
# 고장 종류: "unbalance"(불평형), "looseness"(헐거움), "bearing"(베어링 결함)
FAULTS = [
    dict(kind="unbalance", wheel="right", start_lap=8, end_lap=20, max_severity=1.0),
]

N_LAPS = 20               # 경로 반복 횟수
BASELINE_LAPS = 5         # 정상 기준 학습에 쓸 초기 반복 수
SEED = 7
