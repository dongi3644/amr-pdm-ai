"""실시간 추론기: 1초 창이 들어올 때마다 상태·위험도·원인을 갱신한다.

[역할] 학습이 끝난 bundle을 받아 로봇 1대를 '지켜보는' 객체(Predictor). 로봇마다 하나씩 만든다.
       창 하나는 흔들리기 쉬우므로, 판정은 판정 주기(기본 1분) 단위로 모아서 하고 상태는 2번(기본값) 연속일 때만 바꾼다 (오경보 방지).
[흐름에서 위치] bundle(system.py) + 1초 창(dataio) → (여기) update() → 상태·점수·원인
[누가 쓰나] workflow.diagnose(주기별 표), workflow.cross_validate(평가)

흐름
  (per_unit 기준) 처음 BASELINE_PERIODS주기(기본 5분) → 구간별 정상 기준 학습 (이 로봇만의 '복도 지문')
  (reference 기준) 학습 때 만든 공통 기준을 바로 사용
  이후 창마다 → 진동 특징 → z-score → 오토인코더 건강 점수
  판정 주기가 끝날 때마다 → 주기 점수(기본: 중앙값) → 상태(정상/경미/심각, 기본 2번 연속일 때만 변경)
                      (주기 점수의 지점·연속 횟수는 config.PERIOD_QUANTILE·CONFIRM_PERIODS. 학습 때 값이 모델에 저장됨)
  경미 이상인 동안 → 랜덤포레스트로 원인 확률 누적, 주기마다 SHAP으로 판단 근거 갱신
"""
from collections import deque

import numpy as np

import config as C
from pipeline.features import FEATS, extract
from pipeline.models import NORMAL_LABEL, SegmentBaseline, health, label_ko

FEAT_KO = {"rms": "진동 크기(RMS)", "kurt": "충격성(첨도)", "crest": "크레스트", "low": "저주파 진동",
           "mid": "중주파 진동", "high": "고주파 진동"}


ENC_KO = {**{f"{w}_rel": f"{C.WHEEL_KO[w]} 바퀴 상대 속도" for w in C.WHEEL_KO},
          **{f"{w}_rip": f"{C.WHEEL_KO[w]} 바퀴 속도 흔들림" for w in C.WHEEL_KO},
          "left_diff": "왼쪽 앞뒤 바퀴 속도 차이", "right_diff": "오른쪽 앞뒤 바퀴 속도 차이", "cmd_eff": "모터 명령 대비 속도"}


# ---- 어디가 문제인가: 평소와 가장 다른 특징 → 축·떨림 종류·의심 고장 (11/15) ----
# 축 방향은 센서를 붙인 방향을 따른다 (X 화살표를 로봇 앞쪽으로 붙였다고 보고 쓴 설명)
AXIS_WHERE = {"ax": "X축(앞뒤)", "ay": "Y축(좌우)", "az": "Z축(위아래)",
              "gx": "X축 회전(좌우로 기울어짐)", "gy": "Y축 회전(앞뒤로 끄덕임)", "gz": "Z축 회전(좌우로 돎)"}
KIND_WHERE = {   # 특징 종류 → (무엇이 달라졌나, 의심되는 고장)
    "rms": ("전체 진동 크기", "진동이 전반적으로 커짐 (불균형·풀림·마찰 증가 모두 가능 → 아래 다른 항목과 함께 보기)"),
    "low": ("느린 흔들림(1~20 Hz)", "바퀴가 한 바퀴 돌 때마다 흔들림 → 바퀴 불균형·변형(무게 쏠림)·체결 풀림(나사) 의심"),
    "mid": ("중간 떨림(20~150 Hz)", "모터·기어 회전에 맞춘 떨림 → 모터·기어 이상, 바퀴 축 휨 의심"),
    "high": ("빠른 떨림(150 Hz 이상)", "빠르고 잔 떨림 → 마찰 증가 (베어링 손상·걸림) 의심"),
    "kurt": ("뾰족한 충격", "짧은 '딱' 충격이 반복 → 나사 풀림 덜컹임·기어 이 손상·베어링 흠 의심 (바닥 턱도 확인)"),
    "crest": ("순간 충격 크기", "순간적으로 큰 충격 → 덜컹임·걸림 의심 (바닥 턱도 확인)"),
}


def locate(feats, z_mean, top=3, min_z=2.0):
    """어디가 평소와 가장 다른가 → [{feat, name, z, what, hint}, ...] (|z|가 min_z 이상인 것 중 큰 순서로 top개)
    z_mean: 최근 창들의 특징별 평균 z (평소 대비 몇 배 벗어났나, 표준편차 단위). +면 평소보다 큼, -면 작음.
    진동 센서 하나로는 '어느 바퀴'까지는 모른다 → 축·떨림 종류로 고장 종류를 짐작. 엔코더 4개가 있으면 바퀴 특징이 바퀴를 가리킴"""
    z_mean = np.asarray(z_mean, float)
    out = []
    for i in np.argsort(-np.abs(z_mean)):
        if len(out) >= top or abs(z_mean[i]) < min_z:
            break
        f, z = feats[i], float(z_mean[i])
        up = "평소보다 큼" if z > 0 else "평소보다 작음"
        if f in ENC_KO:                                   # 엔코더 바퀴 특징: 바퀴를 직접 가리킴
            w = f.split("_")[0]
            if f.endswith("_rel"):
                hint = f"{C.WHEEL_KO[w]} 바퀴가 다른 바퀴보다 {'빠름' if z > 0 else '느림'} → 그 바퀴 " + ("헛돎·미끄러짐" if z > 0 else "마찰(베어링)·걸림") + " 의심"
            elif f.endswith("_rip"):
                hint = f"{C.WHEEL_KO[w]} 바퀴만 덜컹이며 돎 → 그 바퀴 체결 풀림·불균형 의심"
            elif f == "cmd_eff":
                hint = "같은 명령에 덜 달림 → 전체 저항 증가(베어링·바닥·짐) 의심"
            else:
                hint = f"{'왼쪽' if f.startswith('left') else '오른쪽'} 앞뒤 바퀴 속도가 달라짐 → 그쪽 바퀴 하나의 마찰·풀림 의심"
            out.append(dict(feat=f, name=ENC_KO[f], z=z, what=up, hint=hint))
            continue
        ch, key = f.split("_")
        what, hint = KIND_WHERE.get(key, (FEAT_KO.get(key, key), ""))
        if z < 0:                                          # 평소보다 작아진 것은 고장보다 센서·주행 조건 변화일 때가 많음
            hint = "평소보다 작아짐 → 센서 고정이 느슨해졌거나 주행 조건(속도·짐)이 바뀌었는지 확인"
        out.append(dict(feat=f, name=f"{AXIS_WHERE.get(ch, ch)} {what}", z=z, what=up, hint=hint))
    return out


def where_text(items):
    """locate() 결과 → 알림에 붙일 한 줄 (가장 다른 곳 하나)"""
    if not items:
        return ""
    a = items[0]
    return f"{a['name']} {abs(a['z']):.1f}배 벗어남 → {a['hint']}"


def feat_ko(f):
    """특징 이름 → 화면에 보일 한국어 이름 (SHAP 판단 근거, 그래프 축에 씀)
    ax_rms → 'X축 가속도 진동 크기(RMS)', gz_high → 'Z축 각속도 고주파 진동'"""
    if f in ENC_KO:                                       # 엔코더 특징 (예: fl_rip → 앞 왼쪽 바퀴 속도 흔들림)
        return ENC_KO[f]
    ch, key = f.split("_")
    return f"{C.CHANNEL_KO.get(ch, ch)} {FEAT_KO.get(key, key)}"


def risk_pct(score):
    """건강 점수 → 고장 위험도 [%] (9점 = 100%, 100에서 멈춤). 점수가 없으면 None"""
    if score is None or not np.isfinite(score):
        return None
    return float(min(100.0, max(0.0, score / C.THRESH["danger"] * 100)))


def stage(status, score, cause_label=None):
    """상태 번호(0 정상 · 1 경미 · 2 심각)와 점수 → (단계 이름, 지금 상황, 할 일)
    단계는 '2분 연속' 규칙을 거쳐 바뀌므로, 이번 1분 점수가 단계와 다르면 상황 칸에 '확인 중'이라고 알려 준다.
    cause_label(원인 분류 결과, 예: 'looseness|fr')을 알면 할 일에 그 고장의 조치(config.FAULT_TODO)를 앞에 붙인다.
    심각이면서 9점(위험도 100%) 이상이면 '학습한 고장 기준 도달 → 운행 멈춤'으로 바꾼다."""
    name = C.STATUS_KO[status]
    situation, todo = C.RISK_TEXT[name]
    if status >= 2 and score is not None and score >= C.THRESH["danger"]:
        situation, todo = "위험도 100%: 학습한 고장 기준(고장 정도 0.8)에 도달", "운행을 멈추고 바로 점검"
    kind = cause_label.split("|")[0] if status >= 1 and cause_label and cause_label != NORMAL_LABEL else None
    if kind in C.FAULT_TODO:
        todo = f"{C.FAULT_TODO[kind]} ({todo})"
    if score is not None and np.isfinite(score):
        now = level(score)                               # 이번 1분 점수만 보면 어느 단계인가
        if now > status:
            situation = f"이번 1분은 {C.STATUS_KO[now]} 수준. 다음 1분도 이어지면 {C.STATUS_KO[now]}로 올라감 (확인 중)"
            todo = f"{todo}. 다음 1분 지켜보기"
        elif now < status:
            situation = f"{situation} (이번 1분은 낮아졌지만 {C.CONFIRM_PERIODS}분 연속이어야 단계가 내려감)"
    return name, situation, todo


def stage_ranges():
    """화면 안내용: 단계마다 위험도 % 범위 문구 → [(이름, '0~33%'), ...]"""
    p = lambda v: round(v / C.THRESH["danger"] * 100)
    t = C.THRESH
    return [("정상", f"0~{p(t['caution'])}% 미만"), ("경미", f"{p(t['caution'])}~{p(t['warning'])}% 미만"),
            ("심각", f"{p(t['warning'])}% 이상 (100% = 학습한 고장 기준)")]


def suspect_part(cause_label, where_items, n_enc):
    """의심 부위 → (바퀴 'fl'·'fr'·'rl'·'rr' 또는 None, 화면 문구)
    ① 평소와 가장 다른 곳(where_items 중 엔코더 바퀴 특징)이 가리키는 바퀴: 그 바퀴만 다르게 도는 것을 직접 잰 근거라 먼저 씀
    ② 없으면 원인 분류가 낸 바퀴 (11/17 가상 시연: 원인 분류는 앞 왼쪽, 엔코더와 실제 고장은 뒤 오른쪽이었던 경우가 있어 순서를 바꿈)
    ③ 바퀴를 모르면: 엔코더가 4개보다 적으면 '구동계 (바퀴 4개 중 특정 불가)' (진동 센서 하나로는 어느 바퀴인지 모름)"""
    for w in where_items or []:
        side = w["feat"].split("_")[0]
        if w["feat"] in ENC_KO and side in C.WHEEL_KO:
            return side, f"{C.WHEEL_KO[side]} 바퀴·모터"
    if cause_label and cause_label != NORMAL_LABEL:
        side = cause_label.split("|")[1] if "|" in cause_label else ""
        if side in C.WHEEL_KO:
            return side, f"{C.WHEEL_KO[side]} 바퀴·모터"
    return None, "구동계 (바퀴 4개 중 특정 불가)" if n_enc < 4 else "구동계 (한 바퀴만 다르게 도는 곳 없음)"


def level(score):
    """건강 점수 → 상태 번호. 3점 미만 0(정상), 6점 미만 1(경미), 그 이상 2(심각)"""
    return 0 if score < C.THRESH["caution"] else 1 if score < C.THRESH["warning"] else 2


class Predictor:
    def __init__(self, bundle, detector=None, use_cause=True):
        # 건강 점수를 낼 모델: 지정하지 않으면 학습 때 정한 주 모델 (엔코더가 있으면 오토인코더 + 바퀴 이상, 없으면 오토인코더)
        self.b, self.det = bundle, bundle["detectors"][detector or bundle.get("main", "ae")]    # 건강 점수를 낼 모델 (기본: 오토인코더)
        self.use_cause = use_cause and bundle["cause"] is not None   # 고장 라벨 없이 학습하면 원인 모델이 없음
        self.calibrated = bundle["scale"] != "normal_only"           # 고장 데이터가 없으면 9점(위험도 100%) 눈금이 없음
        # 주기 점수의 지점·상태 변경 연속 횟수: 학습 때 모델에 저장한 값 (이 값이 없는 예전 모델은 예전 방식 0.5·2번)
        self.quantile = float(bundle.get("period_quantile", 0.5))
        self.confirm = int(bundle.get("confirm_periods", 2))
        self.feats = list(bundle.get("feats", FEATS))     # 모델이 학습한 특징 목록 (엔코더 특징 포함 여부)
        self.skipped = 0                                 # 특징 값이 비어 판정하지 못한 창 수 (예: 엔코더 값이 끊긴 1초)
        self.reset()

    def reset(self):
        """처음 상태로 되돌림 (새 로봇을 지켜보기 시작할 때)"""
        self.base_rows = []                              # per_unit 방식: 처음 5분 동안 모으는 기준용 창
        # 정상 기준. reference 방식이면 학습 때 만든 공통 기준을 바로 쓰고, per_unit이면 5분 뒤에 만든다
        self.baseline = self.b["reference"] if self.b["baseline"] == "reference" else None
        # cur_period: 지금 주기 번호, period_buf: 이번 주기 창 점수들, period_scores: 끝난 주기들의 (번호, 점수)
        self.cur_period, self.period_buf, self.period_scores = None, [], []
        self.status, self.pending = 0, (0, 0)            # pending = (목표 상태, 연속 주기 수)
        # ewma: 화면용 평활 점수, cause_buf: 최근 60개 창의 원인 확률
        self.ewma, self.cause_buf = None, deque(maxlen=C.CAUSE_WINDOW)
        self.period_z, self.reason = [], []              # 이번 주기 창들의 z (SHAP용), 판단 근거 문구
        self.events = []                                 # (주기, 상태) 변경 이력
        self.recent_z = deque(maxlen=30)                 # 최근 30개 창의 z (어디가 평소와 다른가 = where())
        self.last_cause_label = None

    def _finish_period(self, period):
        """판정 주기가 끝날 때 한 번: 주기 점수 → 상태 변경(연속 규칙) → 원인 판단 근거(SHAP) 갱신"""
        if not self.period_buf:
            return
        # 주기 점수 = 그 주기 창 점수들의 quantile 지점. 0.5(중앙값)면 튀는 창 몇 개에 흔들리지 않음
        s = float(np.median(self.period_buf)) if self.quantile == 0.5 else float(np.quantile(self.period_buf, self.quantile))
        self.period_scores.append((period, s))
        self.period_buf = []
        lv = level(s)                                    # 이번 주기 점수만 보면 어떤 상태인가
        if lv != self.status:                            # 지금 상태와 다르면 바로 바꾸지 않고 횟수를 센다
            tgt, cnt = self.pending
            cnt = cnt + 1 if tgt == lv else 1            # 지난 주기와 같은 방향이면 +1, 아니면 1부터 다시
            self.pending = (lv, cnt)
            # confirm번(기본 2번) 연속이면 상태 변경. 단, 9점 이상으로 뛰면 기다리지 않고 바로 올림
            if cnt >= self.confirm or (lv > self.status and s >= C.THRESH["danger"]):
                self.status, self.pending = lv, (lv, 0)
                self.events.append((period, self.status))
                if self.status == 0:                     # 정상 복귀(정비 후): 이전 고장의 원인을 비움
                    self.cause_buf.clear()
                    self.reason = []
        else:
            self.pending = (lv, 0)                       # 지금 상태와 같으면 세던 횟수를 버림
        if self.use_cause and self.status >= 1 and self.cause_buf and self.period_z:
            label = self._top_cause()[0]
            if label != NORMAL_LABEL:                    # 이번 주기 평균 z로 '왜 그 원인인가'를 SHAP으로 계산
                self.reason = [(feat_ko(f), v) for f, v in self.b["cause"].explain(np.mean(self.period_z, axis=0), label)]
        self.period_z = []

    def _top_cause(self):
        """최근 창들의 원인 확률 평균 → (가장 높은 라벨, 확률 배열)"""
        p = np.mean(self.cause_buf, axis=0)
        cls = self.b["cause"].classes
        return cls[int(np.argmax(p))], p

    def current_cause_label(self):
        """지금 상태로 본 추정 원인 라벨 (예: 'looseness|fr'). 경미 이상이고 원인 확률이 쌓였을 때만, 아니면 None"""
        if not (self.use_cause and self.status >= 1 and self.cause_buf):
            return None
        return self._top_cause()[0]

    def current_cause(self):
        """지금 상태로 본 추정 원인 문구 (경미 이상이고 원인 확률이 쌓였을 때만, 아니면 '')
        주기 표는 주기가 끝난 직후(상태가 막 바뀐 직후)에 쓰므로, 마지막 창의 결과가 아니라 이 값을 써야 원인 칸이 한 주기 늦지 않다."""
        if not (self.use_cause and self.status >= 1 and self.cause_buf):
            return ""
        top = self._top_cause()[0]
        return "원인 미확정" if top == NORMAL_LABEL else label_ko(top)

    def update(self, rec):
        """1초 창(dataio의 rec) 하나 처리: 특징을 뽑아 update_row로 넘김"""
        return self.update_row(extract(rec), rec)

    def update_row(self, row, rec=None):
        """창 하나 처리: (기준 학습 중이면 모으기만) → z → 건강 점수 → 평활 점수 → 원인 확률 누적"""
        if row["period"] != self.cur_period:             # 주기 번호가 바뀌었다 = 지난 주기가 끝났다
            if self.cur_period is not None:
                self._finish_period(self.cur_period)
            self.cur_period = row["period"]
        # 이 창의 처리 결과 (화면·표가 쓰는 값). 아래에서 단계마다 채워 넣는다
        out = dict(period=row["period"], segment=row["segment"], row=row, rec=rec, phase="monitor",
                   status=self.status, status_ko=C.STATUS_KO[self.status], score=self.ewma,
                   period_scores=list(self.period_scores), cause=None, cause_probs=None,
                   cause_reason=list(self.reason), top_feats=[], events=list(self.events))

        if self.baseline is None:                         # per_unit 방식: 아직 이 로봇의 기준이 없음
            if row["period"] < C.BASELINE_PERIODS:        # 처음 5분은 창을 모으기만 하고 판정하지 않음
                if not row["bump"] and all(np.isfinite(row[f]) for f in self.feats):   # 값이 빈 창은 기준에 넣지 않음
                    self.base_rows.append(row)
                out["phase"] = "learning"
                return out
            self.baseline = SegmentBaseline().fit(self.base_rows, self.feats)   # 5분이 지나면 모은 창으로 기준을 만듦

        if row["bump"]:                                   # 문턱 통과 순간은 판정에서 제외
            out["phase"] = "skip"
            return out
        if not all(np.isfinite(row[f]) for f in self.feats):   # 모델이 쓰는 특징 값이 비어 있으면 판정하지 않음 (채워 넣지 않음)
            self.skipped += 1
            out["phase"] = "skip"
            return out

        z = self.baseline.z([row])                          # 특징마다 평소 대비 몇 배 벗어났나
        h = float(health(self.det, z)[0])                   # 오토인코더 복원 오차 → 건강 점수 (3 경미 · 6 심각 · 9 = 위험도 100%)
        a = 2 / (C.EWMA_SPAN + 1)                           # 화면 표시용 지수 이동 평균 (최근 약 20개 창)
        self.ewma = h if self.ewma is None else (1 - a) * self.ewma + a * h
        self.period_buf.append(h)                         # 주기가 끝날 때 주기 점수(기본: 중앙값)를 낼 창 점수
        self.period_z.append(z[0])
        self.recent_z.append(z[0])
        # 원인 확률은 점수가 경미 근처(2.4점)까지 올랐거나 이미 경미 이상일 때부터 쌓는다 (정상일 땐 원인을 묻지 않음)
        if self.use_cause and (self.ewma >= 0.8 * C.THRESH["caution"] or self.status >= 1):
            self.cause_buf.append(self.b["cause"].proba(z)[0])
        out.update(score=self.ewma, window_score=h, z=z[0])
        order = np.argsort(-np.abs(z[0]))[:3]             # 평소에서 가장 많이 벗어난 특징 3개
        out["top_feats"] = [(feat_ko(self.feats[i]), float(z[0][i])) for i in order]
        if self.cause_buf and self.status >= 1:           # 경미 이상이면 원인 확률 순위와 추정 원인을 채움
            top, p = self._top_cause()
            cls = self.b["cause"].classes
            out["cause_probs"] = sorted(((label_ko(c), float(v)) for c, v in zip(cls, p)), key=lambda x: -x[1])
            out["cause"] = "원인 미확정" if top == NORMAL_LABEL else label_ko(top)
            out["cause_label"] = self.last_cause_label = top
        return out

    def where(self, top=3, min_z=2.0):
        """최근 30초 동안 평소와 가장 다른 곳 → locate() 결과 (아직 판정 전이면 [])"""
        if not self.recent_z:
            return []
        return locate(self.feats, np.mean(self.recent_z, axis=0), top, min_z)

    def finish(self):
        """기록이 끝났을 때 마지막 주기를 마무리 (주기 번호가 더 바뀌지 않으므로 직접 불러 줘야 함)"""
        if self.cur_period is not None:
            self._finish_period(self.cur_period)
