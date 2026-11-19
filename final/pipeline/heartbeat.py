"""[역할] 지금 무슨 작업 중인지 화면에 알려 준다: 시작할 때 한 번, 그 뒤로는 config.HEARTBEAT_S초(기본 5분)마다 한 번.

    with Heartbeat("학습 중") as hb:     # → [14:03:10] ▶ 학습 중
        ...                              # → [14:08:10] 학습 중 (5분 경과)   ← 5분마다
        hb.set("평가 중")                # → [14:09:02] ▶ 평가 중 (작업이 바뀌면 바로 한 번 알리고 5분 시계도 새로 시작)
                                         # → [14:12:40] ■ 끝: 평가 중 (전체 9.5분)

계산에는 아무 영향이 없다 (따로 도는 스레드가 시각만 보고 글자를 찍음).
"""
import threading
import time
from datetime import datetime

import config as C


def _now():
    return datetime.now().strftime("%H:%M:%S")


class Heartbeat:
    def __init__(self, label, every=None):
        self.label, self.every = label, float(every or C.HEARTBEAT_S)
        self.t0 = self.mark = time.time()                 # 전체 시작 시각, 지금 작업을 시작한 시각
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _loop(self):
        while not self._stop.wait(5):                     # 5초마다 깨어나 5분이 지났는지만 봄
            with self._lock:
                if time.time() - self.mark >= self.every:
                    self.mark = time.time()
                    print(f"[{_now()}] {self.label} ({(time.time() - self.t0) / 60:.0f}분 경과)", flush=True)

    def set(self, label):
        """작업이 바뀌었을 때: 바로 한 번 알리고 5분 시계를 새로 시작"""
        with self._lock:
            self.label, self.mark = label, time.time()
        print(f"[{_now()}] ▶ {label}", flush=True)

    def __enter__(self):
        print(f"[{_now()}] ▶ {self.label}", flush=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        print(f"[{_now()}] ■ 끝: {self.label} (전체 {(time.time() - self.t0) / 60:.1f}분)", flush=True)
        return False
