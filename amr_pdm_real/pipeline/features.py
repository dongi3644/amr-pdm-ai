"""특징 추출: 1초 창의 진동 파형 → 숫자 특징 36개 (+ 엔코더가 있으면 바퀴 특징 11개)

[역할] 수백~수천 개 숫자로 된 진동 파형을, 고장과 관련 있다고 알려진 요약 숫자로 바꾼다. 모든 진동 AI 모델은 이 숫자만 본다.
[흐름에서 위치] dataio의 1초 창 → (여기) → SegmentBaseline(z-score) → 이상 감지·원인 분류
[쓰는 데이터] 차체에 단 MPU6050 하나의 6축 (가속도 ax·ay·az, 각속도 gx·gy·gz).
              속도가 없으므로 회전 주파수(1X) 특징은 계산하지 않는다 (속도를 가정하면 임의 값이 되므로).
              전류·전압은 학습 데이터에 없으므로 여기(진동 AI)에 넣지 않고 power.py가 따로 감시한다.

[특징이 잡아내는 고장 신호]
    RMS(진동 크기)    전반적으로 흔들림이 커짐 — 불평형, 풀림
    첨도·크레스트      뾰족한 충격이 섞임 — 베어링·기어 표면 손상의 초기 신호
    저·중·고주파 에너지 어느 주파수 대역이 커졌는가 — 고장 종류마다 다름
    가속도 vs 각속도   가속도는 '떨림', 각속도는 '차체가 흔들리며 기우는 것' — 같은 이상도 두 센서에 다르게 드러나 서로 보완함

[특징 목록] 축 6개 × 6개 = 36개
    ax_rms, ax_kurt, ax_crest, ax_low, ax_mid, ax_high, ay_rms, ..., gz_high
    저(1~20 Hz)·중(20~150 Hz)·고(150 Hz 이상) 주파수 에너지. 샘플링이 300 Hz보다 낮으면 고주파 특징은 거의 비게 된다
"""
import numpy as np

import config as C

CHANNELS = tuple(C.VIB_CHANNELS)
VIB_KEYS = ["rms", "kurt", "crest", "low", "mid", "high"]


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
ENC_MIN_SPEED = 5.0                                        # 속도로 나눌 때 더하는 값 [펄스/초] (멈춰 있을 때 0으로 나누지 않게)


def enc_features(rec):
    """엔코더 4개(+모터 명령) 1초 → 바퀴 특징 11개. 엔코더가 없거나 값이 모자라면 모두 nan (값을 지어내지 않음)
        {바퀴}_spd   바퀴 속도 [펄스/초, 방향 무시]       — 마찰(베어링)이 생기면 그 바퀴만 느려짐
        {바퀴}_rip   0.1초마다 잰 속도가 다른 세 바퀴와 얼마나 다르게 출렁이나 ÷ 평균 속도 — 풀림·불평형이면 그 바퀴만 덜컹임
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
    spd = {}
    for i, w in enumerate(WHEELS):
        s = abs(cnt[-1, i] - cnt[0, i]) / (t[-1] - t[0])
        spd[w] = s
        out[f"{w}_spd"] = s
        if sub is not None:
            # 다른 세 바퀴(중앙값)와의 차이가 얼마나 출렁이나 → 출발·정지·회전처럼 네 바퀴가 함께 바뀌는 변화는 지워지고
            # 이 바퀴만 덜컹이는 것(풀림·불평형)만 남는다
            others = np.median(np.delete(sub, i, axis=1), axis=1)
            out[f"{w}_rip"] = float(np.std(sub[:, i] - others) / (s + ENC_MIN_SPEED))
    mean = float(np.mean(list(spd.values())))
    out["left_diff"] = (spd["fl"] - spd["rl"]) / (mean + ENC_MIN_SPEED)
    out["right_diff"] = (spd["fr"] - spd["rr"]) / (mean + ENC_MIN_SPEED)
    cmd = rec.get("motor_cmd", float("nan"))
    if not np.isnan(cmd):
        out["cmd_eff"] = mean / abs(cmd) if abs(cmd) > 0 else 0.0
    return out


def extract(rec):
    """1초 창(dataio.load가 만든 dict) → 특징 dict. 라벨은 평가용으로만 옮겨 담고 모델 입력에는 쓰지 않는다.
    진동 36개는 항상, 엔코더 11개는 값을 계산해 두되 실제로 모델에 넣을지는 학습 때 choose_feats가 정한다."""
    row = dict(period=rec["period"], segment=rec["segment"], bump=rec["bump"])
    for ch in CHANNELS:                                    # 축마다 6개 → ax_rms, ax_kurt, ..., gz_high (36개)
        for k, v in vib_features(rec[ch], rec["vib_fs"]).items():
            row[f"{ch}_{k}"] = v
    row.update(enc_features(rec))                          # 바퀴 특징 11개 (엔코더가 없으면 nan)
    row.update(true_kind=rec["true_kind"], true_side=rec["true_side"], true_sev=rec["true_sev"], labeled=rec["labeled"])
    return row


# 모델에 들어가는 진동 특징 이름 36개와 그 순서. 모든 모델이 이 순서대로 숫자를 받으므로 순서를 바꾸면 저장된 모델과 맞지 않게 된다
FEATS = [f"{ch}_{k}" for ch in CHANNELS for k in VIB_KEYS]
# 엔코더 특징 11개 (학습 데이터에 엔코더가 있을 때만 FEATS 뒤에 붙음. 실제로 쓴 목록은 모델 파일의 bundle["feats"])
ENC_BASE = [f"{w}_{k}" for w in WHEELS for k in ("spd", "rip")] + ["left_diff", "right_diff"]
ENC_FEATS = ENC_BASE + ["cmd_eff"]


def choose_feats(rows):
    """학습 창들을 보고 모델에 넣을 특징 목록을 정함 → (특징 목록, 안내 문구)
    config.ENCODER_FEATURES가 'auto'면 창의 90% 이상에 엔코더 값이 있을 때만 엔코더 특징을 넣는다
    (일부 로봇에만 엔코더가 있으면 넣지 않음: 진단할 로봇에 엔코더가 없으면 판정할 수 없게 되므로)"""
    mode = C.ENCODER_FEATURES
    has = lambda f: bool(rows) and np.mean([np.isfinite(r.get(f, np.nan)) for r in rows]) >= 0.9
    if mode is False:
        return list(FEATS), "특징: 진동 36개 (엔코더 특징은 config.ENCODER_FEATURES = False로 끔)"
    if all(has(f) for f in ENC_BASE):
        extra = ENC_BASE + (["cmd_eff"] if has("cmd_eff") else [])
        return list(FEATS) + extra, f"특징: 진동 36개 + 엔코더 {len(extra)}개 (바퀴 속도·속도 흔들림·앞뒤 바퀴 차이" + \
            (", 모터 명령 대비 속도)" if "cmd_eff" in extra else ")")
    if mode is True:
        raise ValueError("config.ENCODER_FEATURES = True인데 학습 데이터의 엔코더 값이 부족합니다 (창의 90% 미만).")
    return list(FEATS), "특징: 진동 36개 (엔코더 4개가 모든 학습 기록에 있지 않아 엔코더 특징은 넣지 않음)"
