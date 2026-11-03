"""시뮬레이션 로봇 1대를 회차별로 진단해 콘솔에 출력한다 (대시보드 없이 빠르게 확인할 때)"""
from _common import run

KIND = "bearing"     # bearing / looseness / unbalance / normal
WHEEL = "right"      # right / left

run(["-u", "run_monitor.py", KIND, WHEEL])
