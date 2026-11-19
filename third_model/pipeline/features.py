"""특징 추출: 1초 창 → 숫자 특징 (11/20 기본: 진동 24 + 바퀴 11 + 주파수 모양 72 + 뾰족함 3 + 1회전 박자 3 = 113개. 코드_파일_설명.md)

[역할] 수백~수천 개 숫자로 된 진동 파형을, 고장과 관련 있다고 알려진 요약 숫자로 바꾼다. 모든 진동 AI 모델은 이 숫자만 본다.
[흐름에서 위치] dataio의 1초 창 → (여기) → SegmentBaseline(z-score) → 이상 감지·원인 분류
[쓰는 데이터] 차체에 단 MPU6050 하나의 6축 (가속도 ax·ay·az, 각속도 gx·gy·gz) + 엔코더 4개·모터 명령(있으면).
              회전 주파수(1X) 진동 특징은 아직 계산하지 않는다.
              전류·전압은 AI 입력에 넣지 않고 power.py가 따로 감시한다.

[특징이 잡아내는 고장 신호]
    RMS(진동 크기)    전반적으로 흔들림이 커짐 — 불평형, 풀림
    첨도·크레스트      뾰족한 충격이 섞임 — 베어링·기어 표면 손상의 초기 신호
    저·중·고주파 에너지 어느 주파수 대역이 커졌는가 — 고장 종류마다 다름
    가속도 vs 각속도   가속도는 '떨림', 각속도는 '차체가 흔들리며 기우는 것' — 같은 이상도 두 센서에 다르게 드러나 서로 보완함

[특징 목록] 진동: 축 6개 × 6개 = 36개 — ax_rms, ax_kurt, ax_crest, ax_low, ax_mid, ax_high, ay_rms, ..., gz_high
           바퀴: 11개 — fl_rel, fl_rip, ..., rr_rip, left_diff, right_diff, cmd_eff (enc_features 참고)
    저(1~20 Hz)·중(20~150 Hz)·고(150 Hz 이상) 주파수 에너지. 샘플링이 300 Hz보다 낮으면 고주파 특징은 거의 비게 된다
"""
import numpy as np

import config as C

ACC = tuple(C.VIB_CHANNELS)                            # 가속도 축 (주파수 모양 특징은 이 축만)
# 진동 특징을 계산할 축: 가속도 + 엔코더로 계산한 gz (config.GYRO_FROM_ENCODERS). gz가 없는 기록이면 그 특징은 nan → 학습 때 빠짐
CHANNELS = ACC + (("gz",) if getattr(C, "GYRO_FROM_ENCODERS", False) and "gz" not in ACC else ())
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high"]
# 주파수 모양 특징 (config.SPEC_FEATURES, 11/19): 칸 경계는 Hz로 고정 (샘플링이 달라도 같은 이름 = 같은 주파수)
SPEC_EDGES = np.geomspace(2.0, 235.0, C.SPEC_BANDS + 1)


def wheel_hz(rec):
    """이 1초의 바퀴 회전 [Hz] = 살아 있는 엔코더 속도(펄스/초)의 중앙값 ÷ 1회전 펄스. 없거나 거의 멈춰 있으면 0"""
    v = [abs(rec.get(k, np.nan)) for k in C.ENCODER_CHANNELS]
    v = [x for x in v if np.isfinite(x) and x > 0]
    fw = float(np.median(v)) / C.ENCODER_PPR if v else 0.0
    return fw if fw > 0.5 else 0.0


def rev_periodicity(x, fs, fw):
    """충격(20 Hz 이상 떨림)의 크기가 바퀴 1회전 간격으로 되풀이되는 정도 → -1~1 (멈춰 있거나 1회전이 창의 절반보다 길면 0)"""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if fw <= 0 or n < 16:
        return 0.0
    lag = int(round(fs / fw))
    if lag < 2 or lag > n // 2:
        return 0.0
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(n, 1 / fs) < 20] = 0               # 느린 흔들림(바퀴 불균형 · 주행)은 빼고 충격만
    e = np.abs(np.fft.irfft(X, n))
    e = np.convolve(e, np.ones(5) / 5, mode="same")       # 충격의 크기(포락선)
    e = e - e.mean()
    a, b = e[:-lag], e[lag:]
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def enc_rev_profile(t, c):
    """(시각, 누적 펄스) 최근 몇 초 → (1차 조화, 나머지 조화) 크기. 각도 칸을 지나는 시간 ÷ 주변 1회전 평균 − 1을 칸별로 평균한 모양의 FFT.
    바퀴가 2바퀴도 안 돌았으면 None"""
    nb, ppr = int(C.ENC_REV_BINS), float(C.ENCODER_PPR)
    c = np.abs(np.asarray(c, float) - c[0])
    t = np.asarray(t, float)
    keep = np.r_[True, np.diff(c) > 0]
    t, c = t[keep], c[keep]
    if len(c) < 10 or c[-1] - c[0] < 2 * ppr:
        return None
    blk = ppr / nb
    lv = np.arange(np.ceil(c[0] / blk), np.floor(c[-1] / blk) + 1) * blk     # 15펄스마다
    du = np.diff(np.interp(lv, c, t))                                         # 그 칸을 지나는 데 걸린 시간
    if len(du) < 2 * nb:
        return None
    loc = np.convolve(du, np.ones(nb) / nb, mode="same")                      # 주변 1회전 평균 (가감속은 지움)
    i = np.arange(len(du))
    ok = (i >= nb // 2) & (i < len(du) - nb // 2) & (loc > 0)
    if ok.sum() < nb:
        return None
    r = du[ok] / loc[ok] - 1
    b = np.round(lv[:-1][ok] / blk).astype(int) % nb                          # 바퀴 각도 칸
    cnt = np.bincount(b, minlength=nb)
    p = np.bincount(b, weights=r, minlength=nb) / np.maximum(cnt, 1)
    F_ = np.fft.rfft(p) / nb * 2
    return float(abs(F_[1])), float(np.sqrt((np.abs(F_[2:]) ** 2).sum()))


def spec_shape(x, fs):
    """진동 파형 1초 → 주파수 칸마다 log10(그 칸 에너지 ÷ 전체 에너지). 전체 크기는 나눠서 없애고 '모양'만 남김"""
    x = np.asarray(x, float) - np.mean(x)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    e = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in zip(SPEC_EDGES[:-1], SPEC_EDGES[1:])])
    return np.log10(e / (e.sum() + 1e-12) + 1e-6)


def vib_features(x, fs):
    """진동 파형 1초 → 6개. x: 파형, fs: 샘플링 [Hz]"""
    x = np.asarray(x, float) - np.mean(x)              # 중력·센서 오프셋·일정한 회전(평균)을 빼고 흔들림만 남김
    ms = np.mean(x ** 2) + 1e-12                       # 평균 제곱 (진동 에너지). 아주 작은 수를 더해 0으로 나누는 것을 막음
    spec = np.abs(np.fft.rfft(x)) / len(x) * 2         # FFT 진폭 스펙트럼 (1초 창이라 주파수 간격 1 Hz)
    f = np.fft.rfftfreq(len(x), 1 / fs)                # spec의 칸마다 몇 Hz인가 (0, 1, 2, ... Hz)
    return dict(
        rms=float(np.sqrt(ms)),                        # 진동 크기: 평균 제곱의 제곱근
        kurt=float(np.mean(x ** 4) / ms ** 2),         # 첨도: 고르게 흔들리면 약 3, 뾰족한 충격이 섞이면 커짐
        crest=float(np.max(np.abs(x)) / np.sqrt(ms)),  # 크레스트: 가장 큰 순간값이 평균 크기의 몇 배인가
        low=float(spec[(f >= 1) & (f < 20)].sum()),    # 1~20 Hz 진폭 합 (느린 흔들림. 주로 불평형·풀림)
        mid=float(spec[(f >= 20) & (f < 150)].sum()),  # 20~150 Hz 진폭 합
        high=float(spec[f >= 150].sum()),              # 150 Hz 이상 진폭 합 (빠른 떨림. 주로 베어링 손상)
    )


WHEELS = [ch.split("_")[1] for ch in C.ENCODER_CHANNELS]   # fl, fr, rl, rr
ENC_MOVING = 30.0                                          # 0.1초 구간의 네 바퀴 속도 중앙값이 이보다 크면 '바퀴가 도는 구간' [펄스/초]


def enc_features(rec):
    """엔코더 4개(+모터 명령) 1초 → 바퀴 특징 11개. 엔코더가 없거나 값이 모자라면 모두 nan (값을 지어내지 않음)
        {바퀴}_rel   다른 세 바퀴보다 얼마나 빠르거나 느린가 (도는 구간, 비율) — 마찰(베어링)이 생기면 그 바퀴만 느려짐
                    (바퀴 속도 자체는 주행·정지·회전에 따라 크게 바뀌어 고장과 섞이므로, 다른 바퀴와의 비율만 씀)
        {바퀴}_rip   0.1초마다 잰 속도가 다른 세 바퀴와 얼마나 다르게 출렁이나 ÷ 평균 속도 (도는 구간만) — 풀림·불평형이면 그 바퀴만 덜컹임
        left_diff   왼쪽 앞·뒤 바퀴 속도 차이 ÷ 평균 속도 — 같은 쪽 바퀴는 같은 명령으로 돌므로 평소엔 0 근처
        right_diff  오른쪽 앞·뒤 바퀴 속도 차이 ÷ 평균 속도
        cmd_eff     모터 명령 대비 평균 속도 — 같은 명령에 덜 달리면 전체 저항이 커진 것 (모터 명령 열이 있을 때만)"""
    out = {f: float("nan") for f in ENC_FEATS}
    t, cnt = rec.get("enc_t"), rec.get("enc_cnt")
    if t is None or cnt is None or len(t) < 3 or t[-1] - t[0] < 0.5:
        return out
    t = np.asarray(t, float)
    cnt = np.asarray(cnt, float)
    edges = np.arange(t[0], t[-1] + 1e-9, 0.1)             # 0.1초 간격
    # 0.1초마다 바퀴별 속도 크기 (구간 수 × 바퀴 4)
    sub = np.abs(np.diff(np.column_stack([np.interp(edges, t, cnt[:, i]) for i in range(len(WHEELS))]), axis=0)) / 0.1 \
        if len(edges) >= 3 else None
    # 흔들림·앞뒤 차이·명령 대비 속도는 '바퀴가 실제로 도는 0.1초 구간'만으로 계산한다.
    # 멈추거나 막 출발하는 구간까지 넣으면 속도가 0 근처라 나눗셈이 튀어서, 정상인데도 크게 벗어난 값이 나온다 (가상 시험에서 확인)
    med = np.median(sub, axis=1) if sub is not None else np.array([])   # 0.1초마다 네 바퀴 속도의 중앙값
    moving = med > ENC_MOVING
    if moving.sum() < 3:                                   # 이 1초 동안 거의 멈춰 있었음 → 흔들림·차이가 생길 수 없으므로 0
        for f in [f"{w}_{k}" for w in WHEELS for k in ("rel", "rip")] + ["left_diff", "right_diff"]:
            out[f] = 0.0
        level = 0.0
    else:
        S, level = sub[moving], float(np.mean(med[moving]))   # 도는 구간의 바퀴별 속도, 그 구간의 평균 속도
        for i, w in enumerate(WHEELS):
            # 다른 세 바퀴(중앙값)와의 차이가 얼마나 출렁이나 → 가감속·회전처럼 네 바퀴가 함께 바뀌는 변화는 지워지고
            # 이 바퀴만 덜컹이는 것(풀림·불평형)만 남는다
            others = np.median(np.delete(S, i, axis=1), axis=1)
            out[f"{w}_rip"] = float(np.std(S[:, i] - others) / level)
            out[f"{w}_rel"] = float(np.mean(S[:, i] - others) / level)   # +면 다른 바퀴보다 빠름, -면 느림
        out["left_diff"] = float(np.mean(S[:, 0] - S[:, 2]) / level)    # 앞 왼쪽 - 뒤 왼쪽
        out["right_diff"] = float(np.mean(S[:, 1] - S[:, 3]) / level)   # 앞 오른쪽 - 뒤 오른쪽
    rv = rec.get("enc_rev")                                # 엔코더 1회전 속도 모양 (dataio가 최근 몇 초로 계산, 바퀴 × (1차, 나머지))
    if getattr(C, "ENC_REV", False) and rv is not None:
        rv = np.asarray(rv, float)
        for i, w in enumerate(WHEELS):
            for j, k in enumerate(("rv1", "rv2")):
                v = rv[:, j]
                others = np.delete(v, i)
                out[f"{w}_{k}"] = float(v[i] - np.median(others)) if np.isfinite(v).all() else 0.0   # 2바퀴도 안 돌았으면 0
    cmd = rec.get("motor_cmd", float("nan"))
    if not np.isnan(cmd):
        out["cmd_eff"] = level / abs(cmd) if abs(cmd) > 0 else 0.0
    return out


def part_enc_features(rec):
    """엔코더가 4개 미만일 때(예: enc2 하나) 쓰는 엔코더별 특징 + 모터 명령. 값이 없으면 nan (값을 지어내지 않음)
        {바퀴}_speff  그 바퀴의 평균 속도 ÷ 모터 명령 — 같은 명령에 덜 달리면 그 바퀴 쪽 저항(마찰·베어링)이 커진 것
        {바퀴}_sprip  0.1초마다 잰 속도에서 1초 동안의 직선 추세(가감속)를 뺀 출렁임 ÷ 평균 속도 — 풀림·불평형이면 덜컹임
        cmd           모터 명령 값 (속도 단계). 속도가 빨라 커진 진동을 고장과 구분하는 데 씀
    바퀴가 실제로 도는 0.1초 구간만 쓴다 (멈추거나 막 출발할 때 0 근처 속도로 나누면 값이 튐)"""
    out = {f: float("nan") for f in PART_FEATS}
    cmd = rec.get("motor_cmd", float("nan"))
    if np.isfinite(cmd):
        out["cmd"] = abs(float(cmd))
    for k, w in zip(C.ENCODER_CHANNELS, WHEELS):
        t, c = rec.get(f"{k}_t"), rec.get(f"{k}_c")
        if t is None or c is None or len(t) < 3 or t[-1] - t[0] < 0.5:
            continue
        t, c = np.asarray(t, float), np.asarray(c, float)
        edges = np.arange(t[0], t[-1] + 1e-9, 0.1)
        if len(edges) < 3:
            continue
        sp = np.abs(np.diff(np.interp(edges, t, c))) / 0.1          # 0.1초마다 속도 크기 [펄스/초]
        moving = sp > ENC_MOVING
        if moving.sum() < 3:                                         # 거의 멈춰 있었음
            level = 0.0
            out[f"{w}_sprip"] = 0.0
        else:
            x, y = np.flatnonzero(moving).astype(float), sp[moving]
            level = float(np.mean(y))
            trend = np.polyval(np.polyfit(x, y, 1), x)               # 1초 안의 가감속(직선 추세)은 빼고 출렁임만 봄
            out[f"{w}_sprip"] = float(np.std(y - trend) / level)
        if np.isfinite(cmd):
            out[f"{w}_speff"] = level / abs(cmd) if abs(cmd) > 0 else 0.0
    return out


def extract(rec):
    """1초 창(dataio.load가 만든 dict) → 특징 dict. 라벨은 평가용으로만 옮겨 담고 모델 입력에는 쓰지 않는다.
    진동 36개는 항상, 엔코더 11개는 값을 계산해 두되 실제로 모델에 넣을지는 학습 때 choose_feats가 정한다."""
    row = dict(period=rec["period"], segment=rec["segment"], bump=rec["bump"])
    for ch in CHANNELS:                                    # 축마다 6개 → ax_rms, ax_kurt, ..., gz_high
        if ch not in rec:                                  # 엔코더로 gz를 계산할 수 없는 기록 → nan (학습 때 choose_feats가 뺌)
            row.update({f"{ch}_{k}": float("nan") for k in VIB_KEYS})
            continue
        for k, v in vib_features(rec[ch], rec["vib_fs"]).items():
            row[f"{ch}_{k}"] = v
    if C.SPEC_FEATURES:                                    # 주파수 모양 (축마다 SPEC_BANDS칸)
        for ch in ACC:
            for i, v in enumerate(spec_shape(rec[ch], rec["vib_fs"])):
                row[f"{ch}_s{i:02d}"] = float(v)
            if getattr(C, "SPEC_REV", False):              # 1회전 박자: 충격 크기의 자기상관을 바퀴 1회전 간격에서 잼
                row[f"{ch}_rev"] = rev_periodicity(rec[ch], rec["vib_fs"], wheel_hz(rec))
            if C.SPEC_PEAK:                                # 뾰족함: 한 주파수만 솟으면 큼 (기어 순음), 넓게 퍼지면 작음 (마찰 잡음·바닥)
                x = np.asarray(rec[ch], float) - np.mean(rec[ch])
                a_ = np.abs(np.fft.rfft(x * np.hanning(len(x))))
                f_ = np.fft.rfftfreq(len(x), 1 / rec["vib_fs"])
                a_ = a_[f_ >= 5]
                row[f"{ch}_peak"] = float(np.log10(a_.max() / (np.median(a_) + 1e-12) + 1e-12)) if len(a_) >= 3 else float("nan")
    row.update(enc_features(rec))                          # 바퀴 특징 11개 (엔코더가 없으면 nan)
    row.update(part_enc_features(rec))                     # 엔코더가 4개 미만일 때 쓰는 엔코더별 특징 + 모터 명령 (없으면 nan)
    row.update(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"], labeled=rec["labeled"],
               synthetic=bool(rec.get("synthetic", False)))   # 가상 이상 창 표시 (augment.py, 학습 때만)
    if C.DEEP_MODEL:                                       # 1D-CNN용 1초 파형 (가속도 + 엔코더 속도)
        from pipeline.deep import wave_from_rec
        row["_wave"] = wave_from_rec(rec)
    return row


# 모델에 들어가는 진동 특징 이름 36개와 그 순서. 모든 모델이 이 순서대로 숫자를 받으므로 순서를 바꾸면 저장된 모델과 맞지 않게 된다
FEATS = [f"{ch}_{k}" for ch in CHANNELS for k in VIB_KEYS]
SPEC_FEATS = ([f"{ch}_s{i:02d}" for ch in ACC for i in range(C.SPEC_BANDS)]
              + ([f"{ch}_peak" for ch in ACC] if C.SPEC_PEAK else [])
              + ([f"{ch}_rev" for ch in ACC] if getattr(C, "SPEC_REV", False) else [])) if C.SPEC_FEATURES else []
SPEC_KO = {f"{ch}_s{i:02d}": f"{C.CHANNEL_KO.get(ch, ch)} {SPEC_EDGES[i]:.0f}~{SPEC_EDGES[i + 1]:.0f} Hz 비중"
           for ch in ACC for i in range(C.SPEC_BANDS)}
SPEC_KO.update({f"{ch}_peak": f"{C.CHANNEL_KO.get(ch, ch)} 뾰족한 주파수 정도" for ch in ACC})
SPEC_KO.update({f"{ch}_rev": f"{C.CHANNEL_KO.get(ch, ch)} 1회전 박자 충격" for ch in ACC})
# 엔코더 특징 11개 (학습 데이터에 엔코더가 있을 때만 FEATS 뒤에 붙음. 실제로 쓴 목록은 모델 파일의 bundle["feats"])
ENC_BASE = [f"{w}_{k}" for w in WHEELS for k in ("rel", "rip")] + ["left_diff", "right_diff"]
# 엔코더 1회전 속도 모양 8개 (config.ENC_REV): {바퀴}_rv1 완만하게 (불평형 쪽), {바퀴}_rv2 좁게 패임 (찍힘 쪽). 다른 세 바퀴보다 얼마나 큰가
ENC_REV_FEATS = [f"{w}_{k}" for w in WHEELS for k in ("rv1", "rv2")] if getattr(C, "ENC_REV", False) else []
ENC_BASE = ENC_BASE + ENC_REV_FEATS
# 바퀴 하나에만 생기는 이상을 보는 특징 (models.WheelMaxDetector가 이 중 가장 크게 벗어난 값을 봄)
WHEEL_FEATS = list(ENC_BASE)
ENC_FEATS = ENC_BASE + ["cmd_eff"]
# 엔코더가 4개 미만일 때 쓰는 특징 (있는 엔코더만 학습에 들어감) + 모터 명령
PART_ENC_FEATS = [f"{w}_{k}" for w in WHEELS for k in ("speff", "sprip")]
PART_FEATS = PART_ENC_FEATS + ["cmd"]


def choose_feats(rows):
    """학습 창들을 보고 모델에 넣을 특징 목록을 정함 → (특징 목록, 안내 문구)
    config.ENCODER_FEATURES가 'auto'면 창의 90% 이상에 엔코더 값이 있을 때만 엔코더 특징을 넣는다
    (일부 로봇에만 엔코더가 있으면 넣지 않음: 진단할 로봇에 엔코더가 없으면 판정할 수 없게 되므로)"""
    mode = C.ENCODER_FEATURES
    has = lambda f: bool(rows) and np.mean([np.isfinite(r.get(f, np.nan)) for r in rows]) >= 0.9
    vib_f = [f for f in FEATS if rows and all(np.isfinite(r.get(f, np.nan)) for r in rows)]   # 모든 학습 창에 있는 축만 (gz가 없는 기록이 있으면 gz 특징을 뺌)
    used = list(dict.fromkeys(f.split("_")[0] for f in vib_f))
    vib = f"진동 {len(vib_f)}개 ({', '.join(used)})" + (" · gz = 엔코더로 계산한 각속도" if "gz" in used else "")
    cmd = ["cmd"] if C.CMD_FEATURE and has("cmd") else []
    if mode is False:
        return list(vib_f) + cmd, f"특징: {vib}" + (" + 모터 명령" if cmd else "") + " (엔코더 특징은 config.ENCODER_FEATURES = False로 끔)"
    if all(has(f) for f in ENC_BASE):
        extra = ENC_BASE + (["cmd_eff"] if has("cmd_eff") else [])
        return list(vib_f) + extra, f"특징: {vib} + 엔코더 {len(extra)}개 (바퀴 상대 속도·속도 흔들림·앞뒤 바퀴 차이" + \
            (", 모터 명령 대비 속도)" if "cmd_eff" in extra else ")")
    if C.PARTIAL_ENCODER_FEATURES:                         # 엔코더가 4개 미만: 있는 엔코더마다 명령 대비 속도·속도 흔들림
        part = [f for f in PART_ENC_FEATS if has(f)]
        if part or cmd:
            wheels = sorted({f.split("_")[0] for f in part}, key=WHEELS.index)
            desc = (f" + 엔코더 {len(part)}개 ({', '.join(C.WHEEL_KO[w] for w in wheels)} 바퀴: 명령 대비 속도·속도 흔들림)" if part else "") + \
                (" + 모터 명령" if cmd else "")
            return list(vib_f) + part + cmd, f"특징: {vib}{desc}. 엔코더가 4개 미만이라 바퀴끼리 비교하는 특징은 쓰지 않음"
    if mode is True:
        raise ValueError("config.ENCODER_FEATURES = True인데 학습 데이터의 엔코더 값이 부족합니다 (창의 90% 미만).")
    return list(vib_f) + cmd, f"특징: {vib}" + (" + 모터 명령" if cmd else "") + " (엔코더 값이 부족해 엔코더 특징은 넣지 않음)"
