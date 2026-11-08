"""아두이노 기록: USB 시리얼로 들어오는 센서 값 → data 폴더의 CSV (파이프라인이 그대로 읽는 형식)

[역할] arduino/amr_logger/amr_logger.ino가 보내는 '센서의 원래 정수값'을 단위가 있는 값으로 바꿔 CSV로 저장한다.
       단위 변환 상수는 스케치의 센서 설정과 짝이 맞아야 한다 (스케치를 바꾸면 아래 상수도 같이 바꿀 것).
[흐름에서 위치] 아두이노(MPU6050 + INA219) → USB 시리얼 → (여기) → data/*.csv → dataio.load
[누가 쓰나] 실행/0_아두이노_기록.py

아두이노가 보내는 한 줄:  t_us,ax,ay,az,gx,gy,gz,shunt,bus
    t_us        아두이노 micros() [µs]. 약 71분마다 0으로 돌아가므로 여기서 이어 붙인다
    ax..gz      MPU6050 원래 값 (가속도 ±4 g, 각속도 ±500 °/s 설정)
    shunt, bus  INA219 원래 값. 몇 줄에 한 번만 읽으므로 나머지 줄은 빈칸
저장하는 CSV:  time,ax,ay,az,gx,gy,gz,current,voltage,robot[,segment,fault_kind,fault_wheel,fault_severity]
"""
import csv
import time as _time
from datetime import datetime
from pathlib import Path

import config as C

# ---- 단위 변환 (스케치의 설정과 짝) ----
ACC_LSB_PER_G = 8192.0          # MPU6050 ACCEL_CONFIG = 0x08 (±4 g)
GYRO_LSB_PER_DPS = 65.5         # MPU6050 GYRO_CONFIG = 0x08 (±500 °/s)
SHUNT_V_PER_LSB = 10e-6         # INA219 션트 전압 레지스터: 1 = 10 µV
SHUNT_OHM = 0.1                 # INA219 모듈의 션트 저항 (모듈에 'R100'이라 적혀 있으면 0.1 Ω)
BUS_V_PER_LSB = 0.004           # INA219 버스 전압 레지스터: 위 13비트, 1 = 4 mV
HEADER = ["t_us", "ax", "ay", "az", "gx", "gy", "gz", "shunt", "bus"]
WRAP = 2 ** 32                  # micros()가 돌아가는 값 (unsigned 32비트)


class LineConverter:
    """아두이노 한 줄(글자) → CSV 한 줄(dict). 시각을 이어 붙이기 위해 앞 줄의 시각을 기억한다"""

    def __init__(self):
        self.t0 = self.last = None    # 첫 줄의 시각, 앞 줄의 시각 [µs]
        self.offset = 0               # micros()가 0으로 돌아간 횟수 × 2^32
        self.bad = 0                  # 형식이 틀려 버린 줄 수 (연결 직후의 깨진 글자 등)

    def __call__(self, line):
        parts = line.strip().split(",")
        if len(parts) != len(HEADER) or parts[0] == "t_us":   # 머리줄이나 잘린 줄은 버림
            self.bad += parts[0] != "t_us"
            return None
        try:
            t_us = int(parts[0])
            acc = [int(v) / ACC_LSB_PER_G for v in parts[1:4]]
            gyr = [int(v) / GYRO_LSB_PER_DPS for v in parts[4:7]]
            shunt = int(parts[7]) if parts[7] else None
            bus = int(parts[8]) if parts[8] else None
        except ValueError:
            self.bad += 1
            return None
        if self.last is not None and t_us < self.last:     # micros()가 한 바퀴 돌아 0부터 다시 시작
            self.offset += WRAP
        self.last = t_us
        t_abs = t_us + self.offset
        if self.t0 is None:
            self.t0 = t_abs
        row = {"time": round((t_abs - self.t0) / 1e6, 6),
               "ax": round(acc[0], 5), "ay": round(acc[1], 5), "az": round(acc[2], 5),
               "gx": round(gyr[0], 4), "gy": round(gyr[1], 4), "gz": round(gyr[2], 4),
               # 전류 = 션트 전압 ÷ 션트 저항, 전압 = 버스 전압 (아래 3비트는 상태 비트라 버림). 읽지 않은 줄은 빈칸
               "current": round(shunt * SHUNT_V_PER_LSB / SHUNT_OHM, 5) if shunt is not None else "",
               "voltage": round((bus >> 3) * BUS_V_PER_LSB, 3) if bus is not None else ""}
        return row


def list_ports():
    """PC에 연결된 시리얼 포트 목록 (포트 이름을 모를 때 확인용)"""
    from serial.tools import list_ports as lp
    return [f"{p.device} ({p.description})" for p in lp.comports()]


def record(port, robot, minutes=0, baud=500000, segment="", fault_kind="", fault_wheel="", fault_severity="",
           out_dir=C.DATA_DIR, ser=None, progress=print):
    """시리얼을 읽어 CSV로 저장 → 저장한 파일 경로. minutes=0이면 Ctrl+C를 누를 때까지 기록한다.
    ser: 시험할 때 가짜 시리얼을 넣는 자리 (보통은 비워 둠)"""
    if ser is None:
        import serial                                      # pyserial (requirements.txt)
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except serial.SerialException as e:
            ports = list_ports()
            raise SystemExit(f"포트 {port}를 열 수 없습니다 ({e}).\n연결된 포트: {', '.join(ports) or '없음'} — "
                             "아두이노 IDE의 시리얼 모니터가 켜져 있으면 끄고 다시 실행하세요.")
        _time.sleep(2.0)                                   # 포트를 열면 아두이노가 다시 시작하므로 잠깐 기다림
        ser.reset_input_buffer()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{robot}_{datetime.now():%Y%m%d_%H%M%S}"
    path, k = out_dir / f"{stamp}.csv", 1
    while path.exists():                                   # 같은 초에 기록한 파일이 이미 있으면 덮어쓰지 않고 _2, _3 ...을 붙임
        k += 1
        path = out_dir / f"{stamp}_{k}.csv"
    labels = {k: v for k, v in dict(segment=segment, fault_kind=fault_kind, fault_wheel=fault_wheel,
                                     fault_severity=fault_severity).items() if v != ""}   # 적은 라벨만 열로 넣음
    fields = ["time", "ax", "ay", "az", "gx", "gy", "gz", "current", "voltage", "robot"] + list(labels)
    conv, n, start, shown = LineConverter(), 0, _time.time(), 0.0
    last = {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        try:
            while True:
                raw = ser.readline()
                if not raw:                                # 1초 동안 아무것도 안 오면 (가짜 시리얼이 끝났거나 연결 끊김)
                    if getattr(ser, "finished", False):
                        break
                    progress("  아두이노에서 값이 오지 않습니다. 연결과 보드 속도(BAUD)를 확인하세요.")
                    continue
                row = conv(raw.decode("ascii", errors="ignore"))
                if row is None:
                    continue
                w.writerow({**row, "robot": robot, **labels})
                n += 1
                if row["current"] != "":
                    last = row
                if row["time"] - shown >= 5:               # 5초마다 진행 상황
                    shown = row["time"]
                    rate = n / max(row["time"], 1e-9)
                    progress(f"  {row['time'] / 60:5.1f}분 · {n:,}줄 · 약 {rate:.0f} Hz"
                             + (f" · 전류 {last['current']:.3f} A · 전압 {last['voltage']:.2f} V" if last else ""))
                if minutes and row["time"] >= minutes * 60:
                    break
        except KeyboardInterrupt:                          # Ctrl+C로 멈춤
            pass
        finally:
            if hasattr(ser, "close"):
                ser.close()
    progress(f"저장: {path} ({n:,}줄, 버린 줄 {conv.bad}개, 걸린 시간 {_time.time() - start:.0f}초)")
    return path
