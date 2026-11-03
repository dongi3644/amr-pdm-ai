"""2. data 폴더의 CSV 형식 검사 + 정상 기준 방식 확인"""
from _common import run

PATH = "data"
BASELINE = "auto"    # auto / per_unit / reference (README.md)

run(["check", "--path", PATH, "--baseline", BASELINE])
