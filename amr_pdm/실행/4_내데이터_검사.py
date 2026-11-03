"""4. 내 데이터 형식 검사. CSV를 user_data 폴더에 넣고 실행한다. 형식은 DATA_FORMAT.md"""
from _common import run

PATH = "user_data"   # 파일 또는 폴더
LAP_S = None         # lap 열이 없을 때 1회차 길이 [초]. None이면 60초
FS = None            # time 열이 없을 때 진동 샘플링 [Hz]. None이면 1000

args = ["run_custom.py", "check", "--path", PATH]
if LAP_S:
    args += ["--lap-s", LAP_S]
if FS:
    args += ["--fs", FS]
run(args)
