# AMR 고장예지 시뮬레이터 (기본 틀)

배송로봇(AMR)이 경로를 반복 주행할 때 **진동·전류·온도 센서 데이터**를 만들어 내고,
고장을 서서히 주입해 **AI가 언제 이상을 감지하는지** 확인하는 파이썬 시뮬레이터입니다.

## 실행

```bash
pip install numpy matplotlib
python run_demo.py
```

결과는 `output/features.csv`(창별 특징값·점수), `output/result.png`(그래프)에 저장됩니다.

## 구조

| 파일 | 역할 |
|---|---|
| `config.py` | 로봇·모터·센서·경로·고장 시나리오 설정 (대부분 여기만 고치면 됨) |
| `models.py` | 물리 모델: 모터 전류·온도, 노면·구동 진동, 고장 성분 |
| `sensors.py` | 센서 모델: 샘플링 속도, 측정 범위, 잡음, 분해능 |
| `simulator.py` | 경로를 반복 주행하며 1초 창마다 좌·우 바퀴 센서 데이터 생성 |
| `detector.py` | 특징 추출 + 구간별 정상 기준 이상 감지 (좌우 차이 포함) |
| `real_data.py` | 실물 로봇 CSV를 같은 형식으로 불러오는 연결부 |
| `run_demo.py` | 전체 실행 + 결과 저장 |

## 흐름

```
config(시나리오) → simulator(주행·고장) → sensors(실제 센서처럼 변환)
      → detector(특징 → 정상 기준 대비 점수 → 정상/주의/경고) → 결과 CSV·그래프
```

## 바꿔 보기

- 고장 종류: `config.FAULTS`의 `kind`를 `unbalance` / `looseness` / `bearing`으로
- 센서 성능: `config.SENSORS["vib_fs"]`를 1000(MPU6050) → 3200(ADXL345)으로 바꿔 베어링 결함 감지 차이 비교
- 경로·노면: `config.ROUTE`의 구간, 속도, 거칠기, 문턱 수
- AI 모델: `detector.py`의 `BaselineDetector`를 Isolation Forest, 오토인코더, 1D-CNN 등으로 교체

## 다음 단계

1. 실물 로봇 측정값으로 `models.py`의 계수 보정
2. `real_data.py`로 실측 데이터를 같은 파이프라인에 연결
3. 필요하면 ROS2 노드로 감싸 Gazebo 주행과 연동 (진동·전류·온도 토픽 발행)

> 주의: 진동·전류·온도 식은 고장 시 신호 변화 경향을 보여주기 위한 단순 모델입니다.
> 계수는 임의값이므로 실측 데이터로 보정해야 합니다.
