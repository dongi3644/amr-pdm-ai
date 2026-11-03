"""AMR 고장예지 AI (실측 데이터 전용, 진동만 사용) — 설정

[역할] 모든 파일이 'import config as C'로 가져다 쓰는 설정값 모음. 동작을 바꾸고 싶으면 대부분 여기만 고치면 된다.
       예) 바퀴 이름은 WHEELS, 열 이름이 다르면 COLUMN_ALIASES, 경고 기준을 바꾸려면 THRESH

[쓰는 데이터] 바퀴 4개의 진동만 쓴다. 전류·배터리·속도·온도는 없으므로 쓰지 않고,
              없는 값을 기본값이나 다른 바퀴 값으로 채워 넣지도 않는다 (임의로 만든 값 없음).
가상(시뮬레이션) 데이터도 쓰지 않는다. 모든 모델은 data 폴더에 넣은 실측 CSV로만 학습·평가한다.
'설계값'은 판단 규칙이고, '초기값'은 학습 데이터로 다시 맞추는 값이다.

[어디서 쓰나]
    경로·파일 읽기 값   → dataio.py
    특징 값             → features.py, models.py(SegmentBaseline)
    정상 기준·판정 값   → system.py, infer.py
    평가·칼만 값        → workflow.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                  # 실측 CSV를 넣는 곳 (형식: README.md)
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
BUNDLE_PATH = ART_DIR / "model_bundle.joblib"

# ---------------- 바퀴 (4륜, 바퀴마다 진동 센서 하나) ----------------
# CSV 열 이름은 '<바퀴>_vib'. 예) fl_vib, fr_vib, rl_vib, rr_vib (네 열이 모두 있어야 함, 바퀴 이름에 밑줄 _ 은 쓰지 말 것)
WHEELS = ["fl", "fr", "rl", "rr"]
WHEEL_KO = {"fl": "앞 왼쪽", "fr": "앞 오른쪽", "rl": "뒤 왼쪽", "rr": "뒤 오른쪽"}
WHEEL_ALIASES = {                 # 열 이름·fault_wheel 라벨에서 알아듣는 다른 이름 (소문자로 비교)
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
    # 바퀴별 진동 열의 다른 이름(vib_fl, fl_vibration, 앞왼쪽진동 ...)은 dataio가 WHEEL_ALIASES로 자동으로 만든다
    "fault_kind": ["kind", "label", "고장종류"],
    "fault_wheel": ["wheel", "고장바퀴"],
    "fault_severity": ["severity", "고장정도"],
}

# ---------------- 판정 주기 (회차 대신 시간으로 묶음) ----------------
# 정해진 경로를 반복하지 않으므로 '회차'가 없다. 1초 창 점수를 PERIOD_S초씩 묶어 그 중앙값으로 상태를 판정하고,
# 상태 변경(2번 연속 규칙)·남은 시간 예측·정상 기준도 이 주기 단위로 센다. 화면과 표에는 '분'으로 보여 준다.
PERIOD_S = 60.0                   # 판정 주기 [s] (짧으면 빨리 반응하지만 흔들림, 길면 안정적이지만 늦음)

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
