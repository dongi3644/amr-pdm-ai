"""AMR 고장예지 AI (실측 데이터 전용) — 설정

[역할] 모든 파일이 'import config as C'로 가져다 쓰는 설정값 모음. 동작을 바꾸고 싶으면 대부분 여기만 고치면 된다.
       예) 진동 열 이름은 VIB_CHANNELS, 열 이름이 다르면 COLUMN_ALIASES, 경고 기준을 바꾸려면 THRESH

[쓰는 데이터] 아두이노로 기록한 두 센서 (기록 방법: 아두이노/README.md)
    MPU6050 1개 (차체)   가속도 ax·ay·az, 각속도 gx·gy·gz  → 진동 AI (학습 필요)
    INA219 1개           전류 current, 배터리 전압 voltage → 전력 감시 (학습 없이 로봇마다 처음 5분과 비교)
              배터리 잔량 센서는 따로 없다. 배터리는 INA219의 전압과, 전류를 시간에 따라 더한 사용량[mAh]으로 본다.
              속도·온도는 쓰지 않는다. 없는 값을 기본값이나 다른 값으로 채워 넣지 않는다 (임의로 만든 값 없음).
가상(시뮬레이션) 데이터도 쓰지 않는다. 모든 모델은 data 폴더에 넣은 실측 CSV로만 학습·평가한다.
'설계값'은 판단 규칙이고, '초기값'은 학습 데이터로 다시 맞추는 값이다.

[어디서 쓰나]
    경로·파일 읽기 값   → dataio.py
    특징 값             → features.py, models.py(SegmentBaseline)
    정상 기준·판정 값   → system.py, infer.py
    전류·배터리 값      → power.py
    평가·칼만 값        → workflow.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                  # 실측 CSV를 넣는 곳 (형식: README.md)
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
BUNDLE_PATH = ART_DIR / "model_bundle.joblib"

# ---------------- 진동 센서 (MPU6050 1개, 차체에 고정) ----------------
# CSV 열 이름 = 아래 6개 (모두 있어야 함). 가속도 단위 g, 각속도 단위 °/s. 축 방향은 센서를 붙인 방향을 따른다
# (예: 센서의 X 화살표를 로봇 앞쪽으로 붙이면 X = 앞뒤). 채널 이름에 밑줄 _ 은 쓰지 말 것 (특징 이름을 나눌 때 씀)
VIB_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
CHANNEL_KO = {"ax": "X축 가속도", "ay": "Y축 가속도", "az": "Z축 가속도",
              "gx": "X축 각속도", "gy": "Y축 각속도", "gz": "Z축 각속도"}
# 가속도·각속도를 센서 원래 정수값으로 적은 CSV일 때 나눌 값 (가속도 값이 50을 넘으면 원래값으로 보고 자동으로 나눔)
# MPU6050 범위 설정마다 다름: 가속도 ±2g 16384 · ±4g 8192 · ±8g 4096 / 각속도 ±250°/s 131 · ±500°/s 65.5 · ±1000°/s 32.8
RAW_ACCEL_LSB_PER_G = 8192        # 기록 스케치의 가속도 범위 ±4g
RAW_GYRO_LSB_PER_DPS = 65.5       # 기록 스케치의 각속도 범위 ±500°/s (스케치가 범위를 안 바꿨다면 기본값 ±250°/s = 131)
CHANNEL_ALIASES = {               # 열 이름이 이렇게 적혀 있어도 알아들음 (소문자로 비교)
    "ax": ["acc_x", "accel_x", "accx", "a_x"], "ay": ["acc_y", "accel_y", "accy", "a_y"], "az": ["acc_z", "accel_z", "accz", "a_z"],
    "gx": ["gyro_x", "gyr_x", "gyrox", "g_x"], "gy": ["gyro_y", "gyr_y", "gyroy", "g_y"], "gz": ["gyro_z", "gyr_z", "gyroz", "g_z"],
}

# ---------------- 전류·배터리 (INA219 1개, 배터리와 모터 드라이버 사이) ----------------
# current = 로봇 전체 전류 [A], voltage = 배터리 전압 [V]. 둘 다 선택 열이고, 진동보다 드물게 기록해도 된다 (빈칸 허용).
# 학습 데이터에 없어도 된다: 진단할 때 로봇마다 처음 BASELINE_PERIODS주기(5분)를 그 로봇의 '평소'로 삼아 비교한다.
BATTERY_CAPACITY_MAH = 1500       # 배터리 용량 (RC카 2S 리포 7.4V 1500mAh). '용량 대비 사용 %' 계산에 씀
BATTERY_LOW_V = 7.0               # 이보다 낮으면 '배터리 낮음' (2S 리포: 만충 8.4V, 셀당 3.5V = 7.0V, 방전 한계 약 6.4V)
POWER_QUANTILE = 0.995            # 처음 5분 동안 1초 평균 전류의 이 지점(99.5%)을 넘으면 '평소보다 높음'

# ---------------- 바퀴 이름 (fault_wheel 라벨용) ----------------
# 진동 센서는 하나지만, 고장 실험에서 어느 바퀴에 고장을 냈는지 라벨로 적으면 원인 분류가 '종류 + 바퀴'로 배운다
WHEELS = ["fl", "fr", "rl", "rr"]
WHEEL_KO = {"fl": "앞 왼쪽", "fr": "앞 오른쪽", "rl": "뒤 왼쪽", "rr": "뒤 오른쪽"}
WHEEL_ALIASES = {                 # fault_wheel 라벨에서 알아듣는 다른 이름 (소문자로 비교)
    "fl": ["front_left", "frontleft", "front-left", "앞왼쪽", "전좌"],
    "fr": ["front_right", "frontright", "front-right", "앞오른쪽", "전우"],
    "rl": ["rear_left", "rearleft", "rear-left", "back_left", "뒤왼쪽", "후좌"],
    "rr": ["rear_right", "rearright", "rear-right", "back_right", "뒤오른쪽", "후우"],
}

# ---------------- 파일 읽기 ----------------
WINDOW_S = 1.0                    # 분석 창 길이 [s]
DEFAULT_SEGMENT = "전체"          # segment 열이 없을 때 쓰는 구간 이름 (값을 만드는 게 아니라 '구간 구분 없음'이라는 뜻)
MIN_FS = 100.0                    # 이보다 낮은 진동 샘플링은 받지 않음 [Hz]
MIN_WINDOW_FILL = 0.8             # 1초 창에 샘플이 이 비율보다 적으면 창을 버림
MIN_TRAIN_WINDOWS = 100           # 학습에 필요한 최소 정상 창 수
COLUMN_ALIASES = {                # 파일의 열 이름이 달라도 알아듣는 이름 (소문자로 비교)
    "time": ["t", "timestamp", "time_s", "sec", "시간"],
    "robot": ["robot_id", "unit", "로봇"],
    "segment": ["구간", "condition"],
    "bump": ["문턱"],
    # 진동 열의 다른 이름(acc_x, gyro_x ...)은 위 CHANNEL_ALIASES에 있다
    "current": ["current_a", "cur", "amp", "전류"],
    "voltage": ["voltage_v", "volt", "bus_v", "battery_v", "전압"],
    "fault_kind": ["kind", "label", "고장종류"],
    "fault_wheel": ["wheel", "고장바퀴"],
    "fault_severity": ["severity", "고장정도"],
}

# ---------------- 판정 주기 (회차 대신 시간으로 묶음) ----------------
# 정해진 경로를 반복하지 않으므로 '회차'가 없다. 1초 창 점수를 PERIOD_S초씩 묶어 그 주기 점수(기본: 중앙값)로 상태를 판정하고,
# 상태 변경(CONFIRM_PERIODS번 연속 규칙)·남은 시간 예측·정상 기준도 이 주기 단위로 센다. 화면과 표에는 '분'으로 보여 준다.
# 아래 세 값은 학습 때 모델에 저장되어 진단할 때 그대로 쓰인다 (바꾸면 train부터 다시)
PERIOD_S = 60.0                   # 판정 주기 [s] (짧으면 빨리 반응하지만 흔들림, 길면 안정적이지만 늦음)
PERIOD_QUANTILE = 0.5             # 주기 점수 = 그 주기 1초 점수들의 이 지점. 0.5 = 중앙값(튀는 창에 강함),
                                  # 0.75 = 상위 25% 지점(가끔만 나타나는 간헐 고장에 민감하지만 정상 점수도 올라 오경보 위험)
CONFIRM_PERIODS = 2               # 상태가 바뀌려면 몇 주기 연속이어야 하나 (1 = 빠르지만 오경보, 3 = 안정적이지만 늦음)

# ---------------- 정상 기준 (설계값) ----------------
# per_unit : 로봇마다 처음 BASELINE_PERIODS주기(기본 5분)로 그 로봇의 구간별 정상 기준을 만든다 (처음엔 정상인 운행 기록)
# reference: 정상 로봇들을 모아 공통 기준을 만든다 (처음부터 고장인 기록)
# auto     : 처음 BASELINE_PERIODS주기에 고장 라벨이 있는 로봇이 하나라도 있으면 reference
BASELINE_MODE = "auto"
BASELINE_PERIODS = 5

# ---------------- 판정 (설계값) ----------------
KIND_KO = {"normal": "정상", "unbalance": "불평형", "looseness": "체결부 풀림", "bearing": "베어링 결함", "fault": "고장",
           "outer_race": "베어링 외륜 결함", "inner_race": "베어링 내륜 결함", "combined": "베어링 복합 결함"}
CAUSE_MIN_SEVERITY = 0.25         # 원인 분류 학습에 쓰는 최소 고장 정도
FAILURE_SEVERITY = 0.8            # 평가용 '정지 위험' 기준 (라벨의 고장 정도)
FAIL_CAL_BAND = (0.75, 0.85)      # 건강 점수 9에 맞출 고장 정도 구간
THRESH = dict(caution=3.0, warning=6.0, danger=9.0)   # 건강 점수: 정상 99.5% 지점 = 3, 고장 정도 0.8 = 9
MIN_FAIL_RATIO = 1.5              # 고장 기준 점수가 정상 99.5% 지점의 이 배수를 넘을 때만 9점 눈금(로그)을 맞춤 (아니면 눈금이 불안정)
SCORE_MAX = 30.0                  # 건강 점수 상한 (표시·평균이 극단값 하나에 끌려가지 않도록)
STATUS_KO = ["정상", "주의", "경고"]
EWMA_SPAN = 20                    # 실시간 점수 평활 (창 개수)
TREND_PERIODS = 6                 # 추세선 비교 모델이 쓰는 최근 판정 주기 수
CAUSE_WINDOW = 60                 # 원인 확률을 평균 내는 최근 창 개수
SEED = 7

# ---------------- 평가 ----------------
CV_MAX_FOLDS = 5                  # 로봇 단위 교차검증 최대 겹 수. 로봇이 이보다 적으면 한 대씩 빼는 방식

# ---------------- 칼만 필터 (초기값) ----------------
# 학습 데이터에 9점까지 간 고장 기록이 있으면 학습 때 그 기록으로 다시 맞춘다 (pipeline/workflow.py: tune_kalman)
KALMAN = dict(q_level=0.01, q_slope=0.2, r=0.05)
KALMAN_GRID = dict(q_level=[0.01, 0.05], q_slope=[0.05, 0.1, 0.2, 0.5], r=[0.02, 0.05, 0.2])


def minutes(periods):
    """판정 주기 수 → 분 (표·그래프·문구에 쓰는 단위)"""
    return periods * PERIOD_S / 60
