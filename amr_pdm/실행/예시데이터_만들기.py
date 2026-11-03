"""실제 파일이 없을 때: 시뮬레이터로 user_data/sample_*.csv.gz 3개를 만든다 (몇 분 걸림).
실제 데이터를 넣을 때는 sample_로 시작하는 파일을 지울 것."""
from _common import run

LAPS = 30

run(["-u", "run_custom.py", "sample", "--laps", LAPS])
