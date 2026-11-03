"""AMRPipeline: 모든 모델을 하나로 묶은 진단 시스템 (실측 데이터로만 학습)

[역할] models.py의 부품들을 올바른 순서로 학습시키고, 결과를 'bundle'(dict) 하나에 담아 파일 하나로 저장·불러온다.
[흐름에서 위치] workflow.train이 특징 dict 목록을 넘김 → (여기) fit → bundle → infer.Predictor가 이 bundle로 진단

[bundle에 들어 있는 것]
    cur        전류 예측 모델 (CurrentModel)
    bat        배터리 소모 예측 모델 (BatteryModel, battery 열이 없으면 active=False)
    battery    배터리 소모 경고 기준 {threshold: 예상의 몇 배부터 경고, median: 정상의 중앙값, n: 정상 측정 수} (배터리 없으면 None)
    detectors  이상 감지 5종 (각자 건강 점수 눈금 r0·r995·rfail 포함)
    cause      원인 분류 (CauseClassifier, 고장 라벨이 없으면 None)
    baseline   'per_unit' 또는 'reference',  reference  공통 정상 기준 (reference일 때만)
    scale      9점 눈금을 무엇으로 맞췄나 ('band' 고장 정도 0.75~0.85 / 'all_fault' 고장 창 전체 / 'normal_only' 고장 없음)
    kalman     칼만 필터 잡음 값,  fit_units·cal_units  모델 학습용·눈금 보정용 정상 로봇

    특징 추출 → Ridge(전류 잔차) → 구간별 z-score
                                     ├→ 오토인코더(건강 점수) → 칼만 필터(남은 시간)
                                     └→ 랜덤포레스트 + SHAP(원인·판단 근거)

정상 기준은 두 방식 중 하나다 (config.BASELINE_MODE).
    per_unit : 로봇마다 처음 BASELINE_PERIODS주기(기본 5분)로 그 로봇만의 기준을 만든다
    reference: 학습용 정상 로봇들을 모아 공통 기준을 만든다
"""
from pathlib import Path

import joblib
import numpy as np

import config as C
from pipeline.features import FEATS
from pipeline.infer import Predictor
from pipeline.models import (NORMAL_LABEL, BatteryModel, BatteryTracker, CauseClassifier, CurrentModel, SegmentBaseline,
                             calibrate, make_detectors)


def battery_ratios(rows):
    """정상 로봇 기록을 진단 때와 똑같이 BatteryTracker에 흘려 '실제 소모 / 예상 소모' 비율을 모은다.
    1.0이면 학습한 정상과 같고 1.3이면 30% 빨리 닳은 것. 새 칸이 떨어져 새로 잰 값만 모은다."""
    out, units = [], {}
    for r in rows:
        units.setdefault(r["robot"], []).append(r)
    for rr in units.values():
        tr = BatteryTracker()
        for r in rr:                                       # 기록 순서 그대로 (dataio가 시각을 이어 붙여 둠)
            tr.update(r)
            ratio, fresh = tr.ratio()
            if fresh:
                out.append(ratio)
    return out


def unseen_battery_ratios(normal):
    """정상 로봇을 한 대씩 빼고 나머지로 BatteryModel을 학습한 뒤, 뺀 로봇의 소모 비율을 잰다 (로봇 단위 교차검증과 같은 생각).
    로봇마다 원래 소모량이 조금씩 다르므로, '처음 보는 정상 로봇'의 비율 분포로 기준을 정해야 오경보가 나지 않는다.
    정상 로봇이 1대면 그 로봇 자신으로 잰다 (분포가 좁게 나와 BATTERY_MIN_MARGIN이 기준이 되기 쉬움)."""
    units = {}
    for r in normal:
        units.setdefault(r["robot"], []).append(r)
    if len(units) < 2:
        return battery_ratios(normal)
    out = []
    for name, rows in units.items():
        m = BatteryModel().fit([r for n, rr in units.items() if n != name for r in rr if not r["bump"]])
        out += battery_ratios(m.add_residuals([dict(r) for r in rows]))      # 복사본에 예측을 붙여 원래 값은 건드리지 않음
    return out


def battery_limit(ratios):
    """정상 비율들 → 경고 기준. 측정이 충분하면 99.5% 지점, 적으면 평균 + 3 표준편차. 최소 1 + BATTERY_MIN_MARGIN"""
    v = np.array(list(ratios), float)
    if len(v) < 5:
        return None
    lim = float(np.quantile(v, C.BATTERY_ALERT_Q)) if len(v) >= 50 else float(v.mean() + 3 * v.std())
    return dict(threshold=round(max(lim, 1 + C.BATTERY_MIN_MARGIN), 3), median=round(float(np.median(v)), 3), n=int(len(v)))


def per_unit_z(rows):
    """로봇마다 처음 BASELINE_PERIODS주기로 기준을 만들고, 그 이후 창의 z를 반환"""
    out_rows, Z, units = [], [], {}
    for r in rows:
        units.setdefault(r["robot"], []).append(r)
    for rr in units.values():
        base_rows = [r for r in rr if r["period"] < C.BASELINE_PERIODS and not r["bump"]]
        use = [r for r in rr if r["period"] >= C.BASELINE_PERIODS and not r["bump"]]
        if not base_rows or not use:
            continue
        out_rows += use
        Z.append(SegmentBaseline().fit(base_rows).z(use))
    return out_rows, (np.vstack(Z) if Z else np.empty((0, len(FEATS))))


def reference_z(rows, ref):
    use = [r for r in rows if not r["bump"]]
    return use, (ref.z(use) if use else np.empty((0, len(FEATS))))


def split_units(names, cal_share=1 / 3):
    """정상 로봇을 모델 학습용과 눈금 보정용으로 나눈다 (로봇 단위로 나눠야 눈금이 새 로봇에도 맞음)"""
    names = sorted(names)
    if len(names) < 2:
        return names, []
    cal = names[-max(1, round(len(names) * cal_share)):]
    return [n for n in names if n not in cal], cal


class AMRPipeline:
    def __init__(self, bundle=None):
        self.bundle = bundle

    def add_residuals(self, rows):
        """특징 dict에 전류 잔차 3개와 배터리 초과 소모 1개를 붙인다 (unit_z 전에 필요)"""
        self.bundle["cur"].add_residuals(rows)
        self.bundle["bat"].add_residuals(rows)
        return rows

    def unit_z(self, rows):
        """학습된 기준 방식으로 창 z를 계산 (add_residuals로 잔차가 먼저 붙어 있어야 함)"""
        if self.bundle["baseline"] == "reference":
            return reference_z(rows, self.bundle["reference"])
        return per_unit_z(rows)

    def fit(self, normal, fault, baseline="per_unit"):
        """normal: 정상 로봇들의 창(특징 dict), fault: 고장 라벨이 있는 로봇들의 창
        1) 전류·배터리 소모 예측 2) 정상 기준 3) 이상 감지 + 건강 점수 눈금 4) 원인 분류 5) 배터리 경고 기준 순서로 학습"""
        # 1) 전류 예측: 정상 창으로만 학습하고, 모든 창에 바퀴별 잔차 특징(res_fl, res_fr, ...)을 붙인다
        cur = CurrentModel().fit([r for r in normal if not r["bump"]])
        cur.add_residuals(normal)
        cur.add_residuals(fault)
        # 1-2) 배터리 소모 예측: 같은 방식으로 정상 창에서 '정상 소모'를 배우고 초과 소모(bat_res)를 붙인다
        bat = BatteryModel().fit([r for r in normal if not r["bump"]])
        bat.add_residuals(normal)
        bat.add_residuals(fault)

        # 2) 정상 로봇을 나눔: 모델 학습용(fit) / 눈금 보정용(cal). 보정을 학습에 안 쓴 로봇으로 해야
        #    '처음 보는 정상 로봇의 99.5%가 3점 아래'라는 눈금이 실제로 성립한다
        units = sorted({r["robot"] for r in normal})
        fit_units, cal_units = split_units(units)
        fit_rows = [r for r in normal if r["robot"] in fit_units]
        cal_rows = [r for r in normal if r["robot"] in cal_units]
        ref = None
        if baseline == "reference":
            ref = SegmentBaseline().fit(fit_rows)          # 보정용 로봇은 기준에 넣지 않아야 '처음 보는 로봇' 눈금이 됨
            _, Zfit = reference_z(fit_rows, ref)
            _, Zcal = reference_z(cal_rows, ref)
            frows, Zf = reference_z(fault, ref)
        else:
            _, Zfit = per_unit_z(fit_rows)
            _, Zcal = per_unit_z(cal_rows)
            frows, Zf = per_unit_z(fault)
        if not len(Zcal):                                  # 정상 로봇이 1대뿐이면 그 로봇의 뒤쪽 30%를 보정에 씀
            cut = int(len(Zfit) * 0.7)
            Zfit, Zcal = Zfit[:cut], Zfit[cut:]

        # 3) 이상 감지 5종을 정상 z로 학습하고, 건강 점수 눈금을 맞춘다 (3점 = 정상 99.5%, 9점 = 고장 기준 창의 중앙값)
        sev = np.array([r["true_sev"] for r in frows], float)
        Zfail, scale = Zf[(sev >= C.FAIL_CAL_BAND[0]) & (sev <= C.FAIL_CAL_BAND[1])], "band"
        if not len(Zfail):                                 # 고장 정도가 없거나 그 구간에 창이 없으면 고장 창 전체로 맞춤
            Zfail = Zf[sev >= C.CAUSE_MIN_SEVERITY]
            scale = "all_fault" if len(Zfail) else "normal_only"
        dets = make_detectors()
        for d in dets.values():
            calibrate(d.fit(Zfit), Zcal, Zfail)

        # 4) 원인 분류: 고장 창(라벨 = 종류|바퀴) + 보정용 정상 창(라벨 = 정상)으로 랜덤포레스트 학습
        cause = None
        if (sev > 0).any():                                # 고장 초반(CAUSE_MIN_SEVERITY 미만)은 패턴이 모호하므로 제외
            keep = (sev == 0) | (sev >= C.CAUSE_MIN_SEVERITY)
            labels = np.array([NORMAL_LABEL if r["true_sev"] == 0 else f"{r['true_kind']}|{r['true_side']}" for r in frows])
            Zc = np.vstack([Zf[keep], Zcal])
            yc = np.concatenate([labels[keep], [NORMAL_LABEL] * len(Zcal)])
            if len(set(yc)) >= 2:
                cause = CauseClassifier().fit(Zc, yc)

        # 5) 배터리 소모 경고 기준: 정상 로봇을 한 대씩 빼 가며 잰 '처음 보는 정상 로봇'의 소모 비율 분포로 정한다
        battery = None
        if bat.active:
            battery = battery_limit(unseen_battery_ratios(normal))

        self.bundle = dict(cur=cur, bat=bat, battery=battery, detectors=dets, cause=cause, feats=FEATS, version="real-1.1",
                           baseline=baseline, reference=ref, scale=scale, kalman=dict(C.KALMAN), fit_units=fit_units, cal_units=cal_units)
        return self

    def save(self, path=C.BUNDLE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)
        return path

    @classmethod
    def load(cls, path=C.BUNDLE_PATH):
        return cls(joblib.load(Path(path)))

    def monitor(self, detector="ae", use_cause=True):
        """로봇 1대를 실시간으로 지켜보는 추론기"""
        return Predictor(self.bundle, detector, use_cause)
