"""[역할] 실측 데이터로 보정한 가상 로봇 생성기 (11/18)

sim_robots.py는 진동을 사람이 정한 식으로 만든다. 실제 로봇은 그보다 훨씬 많이 흔들리고(앞뒤 0.2~0.35 g),
바닥에 따라 1.6~1.9배 달라진다. 그래서 이 생성기는 **실측 기록을 바탕**으로 하고, 실측에 없는 센서만 만들어 넣는다.

    가속도 3축 : 실측 정상 기록(공3옥상·욧골, 속도별)에서 2~6초 조각을 이어 붙임 (로봇마다 축별 크기 ±15%)
    멈춤      : 실측 정지 기록 조각 (모터 명령 0)
    전류·전압 : 같은 실측 조각의 값
    각속도 3축 : 실측 가속도에서 만듦 (X축 회전 ← 좌우 흔들림, Y축 회전 ← 앞뒤 흔들림, 0.5~30 Hz, 30 °/s per g) — 가정
    엔코더 4개 : 바퀴 회전(속도 5에서 2.25 Hz, 실측 봉우리 2.2~2.34 Hz) × 330펄스, 평소 흔들림 4~5%
    고장
      바퀴 불균형·변형 (실측 보정) : 욧골 속도 5에서 잰 '테이프 고장 − 정상' 진동 증가분
          · 바퀴 1·2·3배 회전 주파수 성분 → 그 바퀴의 회전 위상에 맞춘 사인파
          · 나머지 증가분 → 같은 주파수 모양의 잡음
          · 고장 바퀴 엔코더: 1회전마다 ±12% 속도 흔들림 (실측 앞 오른쪽 엔코더 흔들림 2~4배와 맞음)
      체결 풀림·마찰 증가 (실측 없음 → 가정) : sim_robots의 모양(충격·200 Hz 잡음)을 쓰되,
          고장 정도 1에서 늘어나는 진동 에너지를 실측 불균형과 같게 맞춤
    바퀴 위치 : 실측은 앞 오른쪽. 다른 바퀴는 sim_robots.WHEEL_GAIN 비율로 (가정)
    고장 진행·간헐·급성·복합·약한 고장: sim_robots.complex_spec·severity를 그대로 씀
    헷갈리는 정상: 바닥 바뀜(실측 공3옥상 → 실측 욧골 조각으로 바뀜), 짐 실음, 저전압, 속도 바뀜

실측 파일은 학습·검증·시험용으로 나눠, 시험 로봇은 학습에 안 쓴 실측 조각으로만 만든다 (RealPools.split).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sim_robots as S

FS = 480.0                                   # 실측 진동 샘플링 [Hz]
LSB = 8192.0                                 # 원래 정수값 → g (±4 g)
FW_PER_SPD = 2.25 / 5                        # 모터 명령(spd) 1당 바퀴 회전 [Hz]
PPR = 330
GYRO_PER_G = 30.0
FLOOR_ENC_JIT = {"공3옥상": 0.04, "욧골": 0.05}


# ------------------------------------------------------------ 실측 조각 묶음
class RealPools:
    """실측 CSV → (바닥, 속도)별 조각 묶음. folders: {바닥 이름: [폴더, ...]}"""

    def __init__(self, folders, skip=()):
        self.files, self.idle = {}, []
        for floor, dirs in folders.items():
            for d in dirs:
                for f in sorted(Path(d).glob("*.csv")):
                    if any(k in f.name for k in skip):
                        continue
                    t = pd.read_csv(f)
                    arr = np.column_stack([t["ax"] / LSB, t["ay"] / LSB, t["az"] / LSB, t["mA"] / 1000, t["mV"] / 1000])
                    spd = int(t["spd"].mode()[0])
                    if spd == 0 or t["az"].std() / LSB < 0.02:
                        self.idle.append((f.stem, arr))
                    elif len(t) >= 2000 and t["mA"].max() > 0:
                        self.files.setdefault((floor, spd), []).append((f"{floor}/{f.stem}", arr))

    def split(self, seed):
        """파일 단위로 학습·검증·시험 나누기 → {역할: {(바닥, 속도): [배열...]}}"""
        rng = np.random.default_rng(seed)
        out = {r: {} for r in ("train", "val", "test")}
        self.names = {r: [] for r in out}
        for key, lst in sorted(self.files.items()):
            idx = rng.permutation(len(lst))
            n = len(lst)
            if n >= 8:
                cut = [int(round(n * 0.6)), int(round(n * 0.8))]
            elif n >= 3:
                cut = [n - 2, n - 1] if n <= 5 else [int(np.ceil(n / 2)), int(np.ceil(n / 2)) + 1]
            else:                                          # 1~2개뿐인 묶음(속도 3 등)은 시험에만 씀
                cut = [0, 0]
            parts = {"train": idx[:cut[0]], "val": idx[cut[0]:cut[1]], "test": idx[cut[1]:]}
            for r, ii in parts.items():
                if len(ii):
                    out[r][key] = [lst[i][1] for i in ii]
                    self.names[r] += [lst[i][0] for i in ii]
        return out


# ------------------------------------------------------------ 실측 불균형 고장의 '증가분'
class FaultModel:
    """욧골 속도 5 정상 vs 테이프 고장 → 축별 (회전 1·2·3배 사인파 진폭, 나머지 증가분의 주파수 모양·세기)"""
    NFFT = 1024

    def __init__(self, normal_arrs, fault_arrs):
        f = np.fft.rfftfreq(self.NFFT, 1 / FS)
        self.f = f
        Pn, Pf = self._psd(normal_arrs), self._psd(fault_arrs)
        fw = 5 * FW_PER_SPD
        self.harm, self.shape, self.var = [], [], []
        for a in range(3):
            ex = np.clip(Pf[a] - Pn[a], 0, None)
            amps = []
            for h in (1, 2, 3):
                m = np.abs(f - h * fw) <= 0.35
                p = ex[m].sum()
                amps.append(np.sqrt(2 * p))                # 바퀴 회전에 묶인 성분의 사인파 진폭
                ex[m] = 0
            self.harm.append(amps)
            self.shape.append(ex / max(ex.sum(), 1e-12))
            self.var.append(ex.sum())
        self.total_var = float(sum(self.var) + sum(sum(x ** 2 / 2 for x in v) for v in self.harm))

    def _psd(self, arrs):
        n, w = self.NFFT, np.hanning(self.NFFT)
        P = [[], [], []]
        for x in arrs:
            for a in range(3):
                s = x[:, a] - x[:, a].mean()
                for i in range(0, len(s) - n + 1, n // 2):
                    seg = s[i:i + n] * w
                    P[a].append(np.abs(np.fft.rfft(seg)) ** 2 / (w ** 2).sum() / (n / 2))
        return [np.mean(p, 0) for p in P]                 # 합하면 분산이 되도록 맞춘 값

    def noise(self, rng, n, a):
        """축 a의 '나머지 증가분' 모양을 가진 잡음 n샘플 (분산 = 실측 증가분)"""
        X = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / FS)
        X *= np.sqrt(np.interp(fr, self.f, self.shape[a]))
        x = np.fft.irfft(X, n)
        return x * np.sqrt(self.var[a]) / max(x.std(), 1e-12)


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / FS)
    X[(fr < lo) | (fr > hi)] = 0
    return np.fft.irfft(X, len(x))


class _Stream:
    def __init__(self, arrs, rng):
        self.arrs, self.rng = arrs, rng

    def take(self, n):
        parts, got = [], 0
        while got < n:
            src = self.arrs[self.rng.integers(len(self.arrs))]
            L = min(int(self.rng.uniform(2, 6) * FS), len(src))
            s = self.rng.integers(0, len(src) - L + 1)
            p = src[s:s + L].copy()
            p[:, :3] *= self.rng.uniform(0.95, 1.05)
            if parts:
                k = min(20, L, len(parts[-1]))
                w = np.linspace(0, 1, k)[:, None]
                p[:k, :3] = parts[-1][-k:, :3] * (1 - w) + p[:k, :3] * w
            parts.append(p)
            got += L
        return np.concatenate(parts)[:n]


# ------------------------------------------------------------ 로봇 1대
def make_table(spec, pools, idle, fm):
    """spec(sim_robots.complex_spec + floor·spd) → sim_robots.make_table과 같은 열의 표"""
    rng = np.random.default_rng(spec["seed"])
    n = int(spec["minutes"] * 60 * FS)
    tt = np.arange(n) / FS
    t_min = tt / 60
    # 주행 계획: 15~45초 달리고 2~4초 멈춤. 속도 바뀜·바닥 바뀜은 그 시각부터 다른 묶음
    spd = np.zeros(n, int)
    floor = np.empty(n, object)
    i = int(rng.uniform(0, 3) * FS)
    while i < n:
        L = int(rng.uniform(15, 45) * FS)
        tm = tt[i] / 60
        s_now = spec["spd2"] if spec.get("spd_at") is not None and tm >= spec["spd_at"] else spec["spd"]
        f_now = spec["floor2"] if spec.get("floor_at") is not None and tm >= spec["floor_at"] else spec["floor"]
        spd[i:i + L], floor[i:i + L] = s_now, f_now
        i += L + int(rng.uniform(2, 4) * FS)
    floor[spd == 0] = spec["floor"]
    base = np.zeros((n, 5))
    gain = rng.uniform(0.85, 1.15, 3) * spec.get("gain", 1.0)
    code = spd * 10 + np.array([x == "욧골" for x in floor], int)
    edges = np.flatnonzero(np.diff(np.r_[-1, code, -1]) != 0)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        key = (floor[a_], int(spd[a_]))
        src = idle if spd[a_] == 0 else pools[key]
        base[a_:b_] = _Stream(src, rng).take(b_ - a_)
    base[:, :3] *= gain
    base[:, 2] = base[:, 2] - np.mean(base[:, 2]) + 1.0
    mv = spd > 0
    v_scale = base[:, 4] / 7.9
    fw = FW_PER_SPD * spd * np.clip(v_scale, 0.85, 1.1)
    sig = {a: base[:, j].copy() for j, a in enumerate(["ax", "ay", "az"])}
    cur = base[:, 3].copy()
    volt = base[:, 4].copy() - spec.get("v_drop", 0.0) - 0.01 * t_min
    if spec.get("load_at") is not None:                   # 짐 실음: 흔들림 조금 증가, 전류 증가
        after = (t_min >= spec["load_at"]) & mv
        for a in sig:
            sig[a] = np.where(after, (sig[a] - (a == "az")) * (1 + 0.12 * spec["load_level"]) + (a == "az"), sig[a])
        cur = cur + after * 0.2 * spec["load_level"] * cur

    sev = S.severity(spec, t_min)
    shown = sev * S._gate(rng, n) if spec.get("intermittent") else sev
    faults = []
    if spec["fault"]:
        faults.append((spec["kind"], spec["wheel"], shown * spec["strength"], spec.get("bearing_hz", 200.0)))
        if spec.get("second"):
            s2 = spec["second"]
            faults.append((s2["kind"], s2["wheel"], sev * s2["ratio"] * s2["strength"], 200.0))
    phase = {w: np.cumsum(2 * np.pi * fw / FS) + rng.uniform(0, 2 * np.pi) for w in S.WHEELS}
    k_spd = (spd / 5.0) * mv                              # 속도가 빠를수록 고장 진동도 큼 (속도 5 = 실측)
    for kind, w, s, bhz in faults:
        d, roll, pitch = S.WHEEL_GAIN[w]
        d0, r0, p0 = S.WHEEL_GAIN["fr"]
        axg = {"ax": d / d0 * pitch / p0, "ay": d / d0 * roll / r0, "az": d / d0}
        amp = np.sqrt(np.clip(s, 0, None)) * k_spd       # 진동 에너지 ∝ 고장 정도
        if kind == "unbalance":
            for j, a in enumerate(["ax", "ay", "az"]):
                per = sum(fm.harm[j][h - 1] * np.sin(h * phase[w] + j) for h in (1, 2, 3))
                sig[a] += axg[a] * amp * (per + fm.noise(rng, n, j))
            cur += 0.05 * s * cur
        elif kind == "looseness":
            ring = np.exp(-np.arange(40) / 8) * np.sin(2 * np.pi * 60 * np.arange(40) / FS)
            hits = rng.random(n) < (fw / FS) * 2
            imp = np.zeros(n)
            imp[hits] = 0.5 + rng.random(hits.sum())
            imp = np.convolve(imp, ring)[:n]
            parts = {"ax": imp, "ay": 0.5 * imp, "az": 0.7 * imp}
            tot = sum(p[mv].var() for p in parts.values()) if mv.any() else 1.0
            sc = np.sqrt(fm.total_var / max(tot, 1e-12))
            for a in sig:
                sig[a] += axg[a] * amp * sc * parts[a]
            cur += 0.02 * s * cur
        elif kind == "bearing":
            hf = {a: _band(rng.standard_normal(n), bhz - 15, bhz + 15) for a in sig}
            wt = {"ax": 1.0, "ay": 0.75, "az": 0.75}
            tot = sum((wt[a] * hf[a]).var() for a in sig)
            sc = np.sqrt(fm.total_var / max(tot, 1e-12))
            for a in sig:
                sig[a] += axg[a] * amp * sc * wt[a] * hf[a]
            cur += 0.12 * s * cur
    for a in sig:                                          # 센서 한계 ±4 g
        sig[a] = np.clip(sig[a], -4.0, 4.0 - 1 / LSB)
    gx = GYRO_PER_G * _band(sig["ay"], 0.5, 30) + 0.06 * rng.standard_normal(n)
    gy = GYRO_PER_G * _band(sig["ax"], 0.5, 30) + 0.06 * rng.standard_normal(n)
    gz = 0.5 * GYRO_PER_G * _band(sig["ay"] - sig["ax"], 0.5, 15) + 0.06 * rng.standard_normal(n)

    # 엔코더 4개
    erng = np.random.default_rng(spec["seed"] + 1)
    coarse = np.arange(0, tt[-1] + 0.05, 0.05)
    jit0 = np.array([FLOOR_ENC_JIT.get(x, 0.045) for x in floor], float)
    enc = {}
    for w in S.WHEELS:
        mult, jit = np.ones(n), jit0.copy()
        for kind, fwh, s, _ in faults:
            if fwh != w:
                continue
            if kind == "bearing":
                mult, jit = mult * (1 - 0.15 * s), jit + 0.03 * s
            elif kind == "looseness":
                jit = jit + 0.25 * s
            elif kind == "unbalance":
                mult = mult * (1 + 0.12 * s * np.sin(phase[w]))
        if spec.get("load_at") is not None:
            mult = mult * (1 - (t_min >= spec["load_at"]) * 0.06 * spec["load_level"])
        noise = np.interp(tt, coarse, erng.standard_normal(len(coarse)))
        speed = fw * PPR * mult * (1 + jit * noise) * mv
        cnt = np.floor(np.cumsum(speed / FS)).astype(np.int64)
        enc[w] = ((cnt + 32768) % 65536) - 32768

    q = lambda v, step: np.round(v / step) * step
    faulty = sev > 0
    df = pd.DataFrame({
        "time": tt + rng.normal(0, 15e-6, n),
        "ax": q(sig["ax"], 1 / LSB), "ay": q(sig["ay"], 1 / LSB), "az": q(sig["az"], 1 / LSB),
        "gx": q(gx, 1 / 65.5), "gy": q(gy, 1 / 65.5), "gz": q(gz, 1 / 65.5),
        "current": q(cur, 0.001), "voltage": q(volt, 0.004), "robot": spec["name"],
        "fault_kind": np.where(faulty, spec.get("kind", "normal"), "normal"),
        "fault_wheel": np.where(faulty, spec.get("wheel", ""), ""),
        "fault_severity": np.round(sev, 3)})
    for w in S.WHEELS:
        df[f"enc_{w}"] = enc[w]
    df["motor_cmd"] = spd
    return df
