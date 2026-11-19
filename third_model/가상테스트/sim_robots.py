"""[역할] 코드 점검 전용 가상 로봇 생성기 (발표·성능 자료로 쓰지 않음)

로봇 1대의 기록을 아두이노 기록기가 저장하는 CSV와 같은 열(time, ax~gz, current, voltage, 라벨)의 표로 만든다.
100대를 CSV로 쓰면 수 GB가 되므로 파일로 저장하지 않고 메모리의 표를 그대로 dataio에 넣는다.

만드는 신호 (모두 사람이 정한 식이라 실제 로봇과 다름)
    주행: 전진 7.7초 · 정지 2.3초 · 회전 3.4초 · 후진 4.1초 · 정지 1.9초 반복, 출발·정지 0.25초 가감속
          (복잡한 로봇은 속도 speed·시간 배율 dscale을 로봇마다 다르게)
    정상 진동: 모터 회전(바퀴의 48배) 성분 + 바퀴 2배 성분 + 잡음. 로봇마다 진동 크기·잡음·속도·배터리가 조금씩 다름
    고장 3종류 (움직일 때만, 움직임 크기에 비례)
        불평형     바퀴 1회전 주기의 흔들림
        체결부 풀림 바퀴가 돌 때마다 '딱' 하는 충격
        베어링 결함 200 Hz 부근 고주파 잡음 + 전류 증가 (bearing_hz로 다른 주파수도 만들 수 있음)
    고장 바퀴 위치: 센서(MPU6050)가 차체 앞 왼쪽에 조금 치우쳐 있다고 가정해, 바퀴마다 축별 세기를 다르게 줌.
        이 가정은 단순화한 것이라 '어느 바퀴인가' 결과는 실제와 크게 다를 수 있다.

복잡한 경우 (complex_spec)
    고장: 급성(갑자기 큰 고장) · 간헐(가끔만 나타남) · 복합(두 고장이 함께) · 약한 고장 · 속도가 다른 주행 ·
          베어링 주파수 변형(학습에 없던 고장) · 고장 + 배터리 저전압
    정상인데 헷갈리기 쉬운 경우: 중간에 바닥이 거칠어짐 · 중간에 짐을 실음 · 배터리 저전압
"""
import numpy as np
import pandas as pd

FS = 500                                                   # 진동 샘플링 [Hz] (아두이노 스케치와 같음)
PATTERN = [(7.7, "fwd", 0.8), (2.3, "stop", 0), (3.4, "turn", 0.7), (4.1, "back", 0.8), (1.9, "stop", 0)]
CYCLE = sum(p[0] for p in PATTERN)
RAMP = 0.25
KINDS = ["unbalance", "looseness", "bearing"]
WHEELS = ["fl", "fr", "rl", "rr"]
# 바퀴별 (센서까지 거리에 따른 세기, 좌우 흔들림 축 비중, 앞뒤 흔들림 축 비중)
WHEEL_GAIN = {"fl": (1.25, 1.3, 1.3), "fr": (1.0, 0.7, 1.3), "rl": (0.95, 1.3, 0.7), "rr": (0.75, 0.7, 0.7)}


def drive(tt, speed=1.0, dscale=1.0):
    """샘플마다 (pwm × 가감속, 회전 중인가, 출발 직후인가). speed: pwm 배율, dscale: 상태 시간 배율"""
    x = tt % (CYCLE * dscale)
    pwm, turn, start = np.zeros_like(x), np.zeros_like(x, bool), np.zeros_like(x, bool)
    edge = 0.0
    for dur, st, p in PATTERN:
        dur = dur * dscale
        m = (x >= edge) & (x < edge + dur)
        if st != "stop":
            pos = x[m] - edge
            pwm[m] = min(p * speed, 1.0) * np.clip(np.minimum(pos, dur - pos) / RAMP, 0, 1)
            turn[m] = st == "turn"
            start[m] = pos < 0.1
        edge += dur
    return pwm, turn, start


def random_spec(rng, name, fault, kind=None, wheel=None):
    """로봇 1대의 무작위 설정. fault=True면 고장 종류·바퀴·진행 속도·최종 고장 정도도 무작위로 정함"""
    spec = dict(name=name, fault=bool(fault), seed=int(rng.integers(1 << 31)),
                v0=rng.uniform(7.5, 8.4),                 # 시작 배터리 전압 [V]
                gain=rng.uniform(0.85, 1.15),             # 정상 진동 크기
                noise=rng.uniform(0.8, 1.3),              # 센서 잡음 크기
                fw=rng.uniform(2.7, 3.3),                 # 바퀴 회전 속도 [Hz] (pwm 0.8, 8.4 V 기준)
                phase=rng.uniform(0, CYCLE))              # 주행 패턴의 어느 지점에서 기록을 시작했나
    if not fault:
        spec.update(kind="normal", wheel="", minutes=rng.uniform(10, 16))
        return spec
    onset = rng.uniform(6, 8)                              # 고장 시작 [분] (처음 5분은 정상 기준용이라 그 뒤)
    ramp = rng.uniform(6, 12)                              # 고장이 최종 정도까지 커지는 데 걸리는 시간 [분]
    spec.update(kind=kind or str(rng.choice(KINDS)), wheel=wheel or str(rng.choice(WHEELS)),
                onset=onset, ramp=ramp,
                final=rng.uniform(0.5, 1.0),               # 최종 고장 정도 (0.8 미만이면 '정지 위험'까지 안 감)
                shape=rng.uniform(1.0, 2.0),               # 1이면 일정하게, 2면 갈수록 빠르게 나빠짐
                strength=rng.uniform(0.7, 1.3),            # 같은 고장 정도라도 로봇마다 진동에 드러나는 세기가 다름
                minutes=onset + ramp + rng.uniform(1, 2))  # 최종 정도에 이른 뒤 1~2분 더 기록
    return spec


# 복잡한 경우의 이름 (결과 표에 그대로 나옴)
FAULT_CATS = ["점진 고장", "급성 고장", "간헐 고장", "복합 고장", "약한 고장", "속도 다름", "베어링 변형", "고장+저전압"]
NORMAL_CATS = ["정상", "바닥 바뀜", "짐 실음", "저전압 정상"]


def complex_spec(rng, name, cat, kind=None, wheel=None, extreme=False):
    """복잡한 경우의 로봇 1대 설정. cat: FAULT_CATS 또는 NORMAL_CATS 중 하나.
    extreme=True면 학습 때보다 더 극단적인 값 (시험용: 학습에 없던 조건)"""
    fault = cat in FAULT_CATS
    spec = random_spec(rng, name, fault, kind, wheel)
    spec.update(cat=cat, speed=rng.uniform(0.85, 1.1), dscale=rng.uniform(0.8, 1.25))
    if cat == "속도 다름":
        spec["speed"] = (rng.choice([rng.uniform(0.55, 0.65), rng.uniform(1.2, 1.25)]) if extreme
                         else rng.uniform(0.7, 1.2))
    if cat in ("고장+저전압", "저전압 정상"):
        spec["v0"] = rng.uniform(7.05, 7.25)               # 기록 중에 7.0 V 아래로 내려감 → 바퀴가 느려짐
    if cat == "바닥 바뀜":
        spec.update(floor_at=rng.uniform(7, 10), floor_level=rng.uniform(1.0, 2.5), minutes=rng.uniform(12, 18))
    if cat == "짐 실음":
        spec.update(load_at=rng.uniform(7, 10), load_level=rng.uniform(0.5, 1.0), minutes=rng.uniform(12, 18))
    if not fault:
        return spec
    if cat == "급성 고장":                                  # 볼트가 갑자기 빠지는 것처럼 한 번에 큰 고장
        spec.update(mode="sudden", final=rng.uniform(0.6, 1.0), minutes=spec["onset"] + rng.uniform(4, 8))
    elif cat == "간헐 고장":                                # 고장은 진행되지만 진동에는 가끔만 나타남
        spec.update(intermittent=True, kind=kind or str(rng.choice(["looseness", "bearing", "unbalance"], p=[0.6, 0.2, 0.2])))
    elif cat == "복합 고장":                                # 두 번째 고장이 다른 바퀴에 함께 진행
        k2 = str(rng.choice([k for k in KINDS if k != spec["kind"]]))
        spec["second"] = dict(kind=k2, wheel=str(rng.choice(WHEELS)), ratio=rng.uniform(0.5, 1.0),
                              strength=rng.uniform(0.7, 1.3))
    elif cat == "약한 고장":                                # 끝까지 가볍게만 진행 (정지 위험까지 안 감)
        spec.update(final=rng.uniform(0.3, 0.5), strength=rng.uniform(0.6, 1.0))
    elif cat == "베어링 변형":                              # 학습에 없던 결함 주파수의 베어링 고장
        spec.update(kind="bearing", bearing_hz=float(rng.choice([120.0, 320.0])) if extreme else 200.0)
    return spec


def severity(spec, t_min):
    """시각[분]마다 고장 정도 0~1 (고장 시작 전 0, 시작할 때 0.1에서 최종 정도까지. 급성이면 바로 최종 정도)"""
    if not spec["fault"]:
        return np.zeros_like(t_min)
    if spec.get("mode") == "sudden":
        return np.where(t_min >= spec["onset"], spec["final"], 0.0)
    x = np.clip((t_min - spec["onset"]) / spec["ramp"], 0, 1)
    sev = 0.1 + (spec["final"] - 0.1) * x ** spec["shape"]
    return np.where(t_min >= spec["onset"], sev, 0.0)


def _gate(rng, n):
    """간헐 고장용 켜짐/꺼짐 (켜짐 5~30초, 꺼짐 10~60초 반복) → 샘플마다 0 또는 1"""
    g, i, on = np.zeros(n), int(rng.uniform(0, 30) * FS), False
    while i < n:
        d = int((rng.uniform(5, 30) if on else rng.uniform(10, 60)) * FS)
        g[i:i + d] = 1.0 if on else 0.0
        i, on = i + d, not on
    return g


def _add_fault(sig, kind, wheel, sev, strength, k, mv, pw, fw, rng, R, n, bearing_hz=200.0):
    """고장 신호 하나를 sig(ax~gz, cur)에 더한다. sev: 샘플마다 신호에 드러나는 고장 정도"""
    d, roll, pitch = WHEEL_GAIN[wheel]
    s_ = sev * k * strength * d                            # 고장 신호 세기 (움직일 때만)
    if kind == "unbalance":
        sig["ay"] += 0.08 * s_ * roll * np.sin(pw)
        sig["az"] += 0.06 * s_ * np.sin(pw + 0.5)
        sig["gx"] += 1.5 * s_ * roll * np.sin(pw)
        sig["gy"] += 1.5 * s_ * pitch * np.sin(pw + 1)
        sig["gz"] += 3.0 * s_ * np.sin(pw)
        sig["cur"] += 0.15 * sev * mv
    elif kind == "looseness":
        ring = np.exp(-np.arange(40) / 8) * np.sin(2 * np.pi * 60 * np.arange(40) / FS)
        hits = rng.random(n) < (fw / FS) * 2 * sev         # 바퀴 1회전에 두 번꼴로 '딱' (고장 정도에 비례)
        imp = np.zeros(n)
        imp[hits] = 0.3 * (sev * strength * d)[hits] * (0.5 + rng.random(hits.sum()))
        imp = np.convolve(imp, ring)[:n]
        sig["ax"] += pitch * imp
        sig["az"] += 0.7 * imp + 0.02 * s_ * np.sin(2 * pw)
        sig["gx"] += 4 * roll * imp
        sig["gy"] += 8 * pitch * imp
        sig["cur"] += 0.05 * sev * mv
    elif kind == "bearing":
        hf = np.convolve(R(), np.sin(2 * np.pi * bearing_hz * np.arange(20) / FS) * np.hanning(20), "same")
        sig["ax"] += 0.04 * s_ * pitch * hf
        sig["ay"] += 0.03 * s_ * roll * hf
        sig["az"] += 0.03 * s_ * hf
        sig["cur"] += 0.3 * sev * mv


def make_table(spec):
    """설정 → 기록기 CSV와 같은 열의 표 (전류·전압은 10샘플에 한 번만 값, 나머지 빈칸)"""
    rng = np.random.default_rng(spec["seed"])
    n = int(spec["minutes"] * 60 * FS)
    tt = np.arange(n) / FS
    pwm, turn, start = drive(tt + spec["phase"], spec.get("speed", 1.0), spec.get("dscale", 1.0))
    mv, k = pwm > 0, pwm / 0.8
    v_batt = spec["v0"] - 0.02 * tt / 60                   # 배터리는 1분에 0.02 V씩 떨어짐
    fw = spec["fw"] * pwm * v_batt / 8.4                   # 바퀴 회전 [Hz] (샘플마다)
    pw = np.cumsum(2 * np.pi * fw / FS)                    # 바퀴 회전 위상
    pm = 48 * pw                                           # 모터 회전 위상 (기어비 48)
    g, sn = spec["gain"], 0.004 * spec["noise"]
    R = lambda: rng.standard_normal(n)
    sig = dict(
        ax=sn * R() + g * k * (0.03 * np.sin(pm) + 0.01 * np.sin(2 * pw) + 0.015 * spec["noise"] * R()),
        ay=sn * R() + g * k * (0.02 * np.sin(pm + 1) + 0.012 * spec["noise"] * R()),
        az=1.0 + sn * R() + g * k * (0.025 * np.sin(pm + 2) + 0.02 * spec["noise"] * R()),
        gx=0.06 * R() + g * k * (0.4 * R() + 1.0 * np.sin(pw)),
        gy=0.06 * R() + g * k * 0.4 * R(),
        gz=0.06 * R() + g * k * 0.4 * R() + 60.0 * k * turn)
    sig["cur"] = 0.05 + 0.01 * R() + mv * (0.55 * pwm + 0.25 * turn * k + 0.03 * R()) + 1.5 * start

    t_min = tt / 60
    sev = severity(spec, t_min)                            # 라벨에 적히는 고장 정도
    shown = sev                                            # 진동에 드러나는 고장 정도 (간헐 고장이면 아래에서 바뀜)
    if spec["fault"]:
        shown = sev * _gate(rng, n) if spec.get("intermittent") else sev   # 진동에 드러나는 고장 정도
        _add_fault(sig, spec["kind"], spec["wheel"], shown, spec["strength"], k, mv, pw, fw, rng, R, n,
                   spec.get("bearing_hz", 200.0))
        if spec.get("second"):                             # 복합 고장: 두 번째 고장 (라벨은 첫 번째 고장만)
            s2 = spec["second"]
            _add_fault(sig, s2["kind"], s2["wheel"], sev * s2["ratio"], s2["strength"], k, mv, pw, fw, rng, R, n)
    if spec.get("floor_at") is not None:                   # 거친 바닥: 바퀴와 상관없는 불규칙 충격 + 잔진동 증가
        after = (t_min >= spec["floor_at"]) & mv
        lvl = spec["floor_level"]
        ring = np.exp(-np.arange(30) / 6) * np.sin(2 * np.pi * 35 * np.arange(30) / FS)
        bumps = np.zeros(n)
        hit = after & (rng.random(n) < 2.0 / FS)           # 1초에 2번꼴
        bumps[hit] = 0.08 * lvl * (0.3 + rng.random(hit.sum()))
        bumps = np.convolve(bumps, ring)[:n] * k
        sig["az"] += bumps + after * 0.01 * lvl * k * R()
        sig["ax"] += 0.5 * bumps
        sig["gy"] += 3 * bumps
    if spec.get("load_at") is not None:                    # 짐을 실음: 전류 증가, 차체가 조금 더 출렁임
        after = (t_min >= spec["load_at"]) & mv
        lvl = spec["load_level"]
        sig["cur"] += after * 0.25 * lvl * pwm
        sig["az"] += after * 0.006 * lvl * k * np.sin(pw)
        sig["gy"] += after * 0.3 * lvl * k * R()
    volt = v_batt - 0.15 * sig["cur"]

    # 센서 해상도로 반올림 (MPU6050 ±4g·±500°/s, INA219 전류 1 mA·전압 4 mV)
    q = lambda v, step: np.round(v / step) * step
    every10 = np.arange(n) % 10 == 0                       # 전류·전압은 10샘플에 한 번 (아두이노 스케치와 같음)
    faulty = sev > 0
    df = pd.DataFrame({
        "time": tt + rng.normal(0, 15e-6, n),              # 시각 흔들림 ±15 µs
        "ax": q(sig["ax"], 1 / 8192), "ay": q(sig["ay"], 1 / 8192), "az": q(sig["az"], 1 / 8192),
        "gx": q(sig["gx"], 1 / 65.5), "gy": q(sig["gy"], 1 / 65.5), "gz": q(sig["gz"], 1 / 65.5),
        "current": np.where(every10, q(sig["cur"], 0.001), np.nan), "voltage": np.where(every10, q(volt, 0.004), np.nan),
        "robot": spec["name"],
        "fault_kind": np.where(faulty, spec["kind"], "normal"),
        "fault_wheel": np.where(faulty, spec["wheel"], ""),
        "fault_severity": np.round(sev, 3)})
    if spec.get("glitch"):                                 # 센서 읽기 오류: 1초에 glitch번꼴로 아무 축 하나의 샘플 하나가 엉뚱한 값
        grng = np.random.default_rng(spec["seed"] + 2)       # (실측 첫 주행 기록에서 본 것: Z축 샘플 하나가 -258 원래값으로 떨어짐)
        idx = np.flatnonzero(grng.random(n) < spec["glitch"] / FS)
        axes = ["ax", "ay", "az", "gx", "gy", "gz"]
        for i in idx:
            ch = axes[grng.integers(6)]
            df.at[i, ch] = -258 / 8192 if ch[0] == "a" else -258 / 65.5
    if spec.get("encoders"):                               # 엔코더 4개 + 모터 명령 (다른 난수를 써서, 끄면 예전과 똑같은 기록)
        enc = _encoders(spec, tt, pwm, fw, pw, sev, shown)
        for w in WHEELS:
            df[f"enc_{w}"] = enc[w]
        df["motor_cmd"] = np.round(pwm * 255)
    return df


PPR = 330                                                  # 바퀴 1회전당 엔코더 펄스 (홀 센서 11 × 기어비 30 가정)


def sides(tt, dscale=1.0):
    """샘플마다 왼쪽·오른쪽 바퀴의 회전 방향 (+1 앞으로, -1 뒤로, 0 정지). 제자리 회전은 왼쪽 +, 오른쪽 -"""
    x = tt % (CYCLE * dscale)
    left, right = np.zeros_like(x), np.zeros_like(x)
    edge = 0.0
    for dur, st, _ in PATTERN:
        dur = dur * dscale
        m = (x >= edge) & (x < edge + dur)
        left[m], right[m] = {"fwd": (1, 1), "turn": (1, -1), "back": (-1, -1), "stop": (0, 0)}[st]
        edge += dur
    return left, right


def _encoders(spec, tt, pwm, fw, pw, sev, shown):
    """가상 엔코더 누적 펄스 (16비트 int처럼 넘침). 고장이 난 바퀴만 속도가 바뀐다고 가정 (사람이 정한 식)
        베어링 결함: 마찰로 그 바퀴가 느려짐 (고장 정도 1에서 15%) + 속도가 조금 흔들림
        체결부 풀림: 바퀴가 덜컹이며 속도가 크게 흔들림
        불평형    : 바퀴 1회전마다 속도가 주기적으로 오르내림
        거친 바닥 : 네 바퀴 모두 속도가 흔들림 / 짐 실음: 네 바퀴 모두 조금 느려짐"""
    erng = np.random.default_rng(spec["seed"] + 1)
    n = len(tt)
    left, right = sides(tt + spec["phase"], spec.get("dscale", 1.0))
    t_min = tt / 60
    mult = {w: np.ones(n) for w in WHEELS}                 # 속도 배율
    jit = {w: np.full(n, 0.02) for w in WHEELS}            # 속도 흔들림 크기 (평소 2%)
    if spec.get("floor_at") is not None:
        for w in WHEELS:
            jit[w] = jit[w] + (t_min >= spec["floor_at"]) * 0.04 * spec["floor_level"]
    if spec.get("load_at") is not None:
        for w in WHEELS:
            mult[w] = mult[w] * (1 - (t_min >= spec["load_at"]) * 0.06 * spec["load_level"])
    faults = []
    if spec["fault"]:
        faults.append((spec["kind"], spec["wheel"], shown * spec["strength"]))
        if spec.get("second"):
            s2 = spec["second"]
            faults.append((s2["kind"], s2["wheel"], sev * s2["ratio"] * s2["strength"]))
    for kind, w, s in faults:
        if kind == "bearing":
            mult[w] = mult[w] * (1 - 0.15 * s)
            jit[w] = jit[w] + 0.03 * s
        elif kind == "looseness":
            jit[w] = jit[w] + 0.25 * s
        elif kind == "unbalance":
            mult[w] = mult[w] * (1 + 0.12 * s * np.sin(pw))
    coarse = np.arange(0, tt[-1] + 0.05, 0.05)             # 흔들림은 20 Hz로 바뀌는 잡음 (샘플마다 바뀌면 0.1초 평균에서 사라짐)
    out = {}
    for w in WHEELS:
        noise = np.interp(tt, coarse, erng.standard_normal(len(coarse)))
        side = left if w in ("fl", "rl") else right
        speed = fw * PPR * mult[w] * (1 + jit[w] * noise) * side      # 펄스/초 (fw는 이미 pwm·배터리 전압에 비례)
        cnt = np.floor(np.cumsum(speed / FS)).astype(np.int64)
        out[w] = ((cnt + 32768) % 65536) - 32768                       # 아두이노 int(16비트)처럼 넘침
    return out
