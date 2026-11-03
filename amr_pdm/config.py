"""AMR 고장예지 AI — 전체 설정 (대부분의 값은 여기서만 바꾸면 됨)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

# ---------------- 로봇 · 모터 ----------------
ROBOT = dict(mass=20.0, wheel_radius=0.08, rolling_coef=0.015)
MOTOR = dict(kt=0.35, i_noload=0.25, resistance=2.0, thermal_cap=60.0, thermal_res=8.0, ambient=24.0)

# ---------------- 센서 ----------------
SENSORS = dict(
    vib_fs=1000,          # 진동 [Hz] (MPU6050 근사, 3200이면 ADXL345)
    vib_range_g=4.0, vib_noise_g=0.004,
    cur_fs=100, cur_noise_a=0.03,
    temp_fs=1, temp_res=0.0625,
)
PHYS_FS = 16000           # 내부 물리 신호 생성 주파수 [Hz] (1000·3200 Hz 센서 모두의 정수배)
WINDOW_S = 1.0            # 분석 창 길이 [s]

# ---------------- 회차별 운행 조건 변화 (현실성) ----------------
LAP_LOAD = (0.9, 1.25)            # 적재 무게 배율
LAP_ROUGHNESS = (0.8, 1.25)       # 바닥 상태 배율
AMBIENT_SWING = 2.0               # 주변 온도 변화 폭 [°C]
AMBIENT_PERIOD_LAPS = 40          # 주변 온도 변화 주기 [회차]

# ---------------- 주행 경로 (구간별 노면) ----------------
ROUTE = [
    dict(name="로비", length=12, speed=0.8, roughness=0.10, bumps=0),
    dict(name="복도A", length=20, speed=1.0, roughness=0.15, bumps=1),
    dict(name="엘리베이터 앞", length=6, speed=0.5, roughness=0.30, bumps=2),
    dict(name="복도B", length=18, speed=1.0, roughness=0.20, bumps=0),
]

# ---------------- 학습 · 판정 ----------------
BASELINE_LAPS = 5                 # 로봇마다 처음 N회 주행으로 구간별 정상 기준 학습 (복도 지문)
FAULT_KINDS = ["unbalance", "looseness", "bearing"]
KIND_KO = {"normal": "정상", "unbalance": "불평형", "looseness": "체결부 풀림", "bearing": "베어링 결함"}
SIDE_KO = {"left": "좌측", "right": "우측"}
CAUSE_MIN_SEVERITY = 0.25         # 원인 분류 학습에 쓰는 최소 고장 정도
FAILURE_SEVERITY = 0.8            # 평가용 '정지 위험' 기준 (실제 고장 정도)
FAIL_CAL_BAND = (0.75, 0.85)      # 건강 점수 9에 맞출 고장 정도 구간

# 건강 점수(0~): 정상 데이터의 99.5% 지점 = 3, 고장 정도 0.8 지점 = 9 (그 사이는 로그 눈금)
THRESH = dict(caution=3.0, warning=6.0, danger=9.0)
STATUS_KO = ["정상", "주의", "경고"]
EWMA_SPAN = 20                    # 실시간 점수 평활 (창 개수)
TREND_LAPS = 6                    # 추세선 비교 모델이 쓰는 최근 회차 수
KALMAN = dict(q_level=0.01, q_slope=0.2, r=0.05)    # 칼만 필터 잡음 (학습 로봇으로 조정)
CAUSE_WINDOW = 60                 # 원인 확률을 평균 내는 최근 창 개수
SEED = 7

# ---------------- 사용자 데이터 (형식 설명: DATA_FORMAT.md) ----------------
USER_DATA_DIR = ROOT / "user_data"          # 여기에 CSV를 넣으면 자동으로 읽음
USER_REPORT_DIR = REPORT_DIR / "user"
USER_BUNDLE_PATH = ART_DIR / "user_model.joblib"
USER_LAP_S = 60.0                 # lap 열이 없을 때 1회차로 볼 시간 [s]
USER_SPEED = 1.0                  # speed 열이 없을 때 가정할 속도 [m/s]
USER_SEGMENT = "전체"             # segment 열이 없을 때 쓰는 구간 이름
USER_MIN_FS = 100.0               # 이보다 낮은 진동 샘플링은 받지 않음 [Hz]
USER_MIN_WINDOW_FILL = 0.8        # 1초 창에 샘플이 이 비율보다 적으면 창을 버림
USER_MIN_TRAIN_WINDOWS = 100      # 학습에 필요한 최소 정상 창 수 (기준 학습 구간 제외)
COLUMN_ALIASES = {                # 파일의 열 이름이 달라도 알아듣는 이름 (소문자로 비교)
    "time": ["t", "timestamp", "time_s", "sec", "시간"],
    "robot": ["robot_id", "로봇"],
    "lap": ["회차"],
    "segment": ["구간"],
    "speed": ["speed_mps", "velocity", "속도"],
    "bump": ["문턱"],
    "left_vib": ["vib_left", "left_vibration", "vibration_left", "vib", "vibration", "좌측진동"],
    "right_vib": ["vib_right", "right_vibration", "vibration_right", "우측진동"],
    "left_cur": ["cur_left", "left_current", "current_left", "current", "좌측전류"],
    "right_cur": ["cur_right", "right_current", "current_right", "우측전류"],
    "left_temp": ["temp_left", "left_temperature", "temperature_left", "temp", "temperature", "좌측온도"],
    "right_temp": ["temp_right", "right_temperature", "temperature_right", "우측온도"],
    "fault_kind": ["kind", "label", "고장종류"],
    "fault_wheel": ["wheel", "고장바퀴"],
    "fault_severity": ["severity", "고장정도"],
}
