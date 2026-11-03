"""공개 실측 데이터: Paderborn 대학 베어링 데이터를 내려받아(베어링당 약 160 MB) data/ 의 CSV로 바꾼다.
라이선스 CC BY-NC 4.0 — 비상업 학술 이용 가능, 발표 자료에 출처 표기 필수 (README.md)."""
from _common import run

# 정상 6개 전부 + 가속 수명 시험으로 생긴 실제 손상 5개 (외륜 2·내륜 2·복합 1), 약 1.8 GB
# 공통 정상 기준은 정상 베어링이 많아야 처음 보는 정상 베어링을 오경보하지 않으므로 정상은 6개 모두 받는다
BEARINGS = "K001,K002,K003,K004,K005,K006,KA04,KA15,KI04,KI16,KB23"
TARGET_FS = 8000     # 64 kHz 진동을 이 주파수로 낮춰 저장 (앨리어싱 방지 필터 포함)

run(["paderborn", "download", "--bearings", BEARINGS], last=False)
run(["paderborn", "convert", "--bearings", BEARINGS, "--target-fs", TARGET_FS])
