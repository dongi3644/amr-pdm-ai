"""2. data 폴더의 CSV 형식 검사 + 정상 기준 방식 확인"""
from _common import run

PATH = "data"        # CSV가 있는 폴더 (다른 폴더나 파일 하나를 검사하려면 여기를 바꿈)
BASELINE = "auto"    # auto / per_unit / reference (README.md)

run(["check", "--path", PATH, "--baseline", BASELINE])
