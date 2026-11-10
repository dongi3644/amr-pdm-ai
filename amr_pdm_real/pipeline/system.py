"""AMRPipeline: 모든 모델을 하나로 묶은 진단 시스템 (실측 진동 데이터로만 학습)

[역할] models.py의 부품들을 올바른 순서로 학습시키고, 결과를 'bundle'(dict) 하나에 담아 파일 하나로 저장·불러온다.
[흐름에서 위치] workflow.train이 특징 dict 목록을 넘김 → (여기) fit → bundle → infer.Predictor가 이 bundle로 진단

[bundle에 들어 있는 것]
    detectors  이상 감지 5종 (각자 건강 점수 눈금 r0·r995·rfail 포함)
    cause      원인 분류 (CauseClassifier, 고장 라벨이 없으면 None)
    baseline   'per_unit' 또는 'reference',  reference  공통 정상 기준 (reference일 때만)
    scale      9점 눈금을 무엇으로 맞췄나 ('band' 고장 정도 0.75~0.85 / 'all_fault' 고장 창 전체 / 'normal_only' 고장 없음)
    kalman     칼만 필터 잡음 값,  fit_units·cal_units  모델 학습용·눈금 보정용 정상 로봇,  period_s  판정 주기

    진동 특징 → 구간별 z-score
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
from pipeline.features import FEATS, WHEEL_FEATS
from pipeline.infer import Predictor
from pipeline.models import NORMAL_LABEL, CauseClassifier, CombinedDetector, SegmentBaseline, WheelMaxDetector, calibrate, make_detectors


def usable(rows, feats):
    """특징 값이 하나라도 비어 있는(nan) 창은 뺌 (예: 엔코더 값이 끊긴 1초). 빈 값을 다른 값으로 채우지 않는다"""
    return [r for r in rows if all(np.isfinite(r[f]) for f in feats)]


def per_unit_z(rows, feats=FEATS):
    """로봇마다 처음 BASELINE_PERIODS주기로 기준을 만들고, 그 이후 창의 z를 반환"""
    out_rows, Z, units = [], [], {}
    for r in usable(rows, feats):                                         # 창들을 로봇별로 모음
        units.setdefault(r["robot"], []).append(r)
    for rr in units.values():
        base_rows = [r for r in rr if r["period"] < C.BASELINE_PERIODS and not r["bump"]]   # 처음 5분 = 기준 만들 창
        use = [r for r in rr if r["period"] >= C.BASELINE_PERIODS and not r["bump"]]        # 그 뒤 = 점수 매길 창
        if not base_rows or not use:                       # 기록이 5분보다 짧은 로봇은 건너뜀
            continue
        out_rows += use
        Z.append(SegmentBaseline().fit(base_rows, feats).z(use))
    return out_rows, (np.vstack(Z) if Z else np.empty((0, len(feats))))


def reference_z(rows, ref):
    """공통 기준(ref) 하나로 모든 창의 z를 계산 (처음 5분도 버리지 않음)"""
    feats = getattr(ref, "feats", FEATS)
    use = [r for r in usable(rows, feats) if not r["bump"]]
    return use, (ref.z(use) if use else np.empty((0, len(feats))))


def split_units(names, cal_share=1 / 3):
    """정상 로봇을 모델 학습용과 눈금 보정용으로 나눈다 (로봇 단위로 나눠야 눈금이 새 로봇에도 맞음)"""
    names = sorted(names)                                  # 이름순으로 정렬해 매번 같은 로봇이 같은 쪽에 가게 함
    if len(names) < 2:                                     # 1대뿐이면 나눌 수 없음 (fit에서 그 로봇의 뒤 30%를 보정에 씀)
        return names, []
    cal = names[-max(1, round(len(names) * cal_share)):]   # 뒤쪽 1/3(최소 1대) = 눈금 보정용
    return [n for n in names if n not in cal], cal


class AMRPipeline:
    def __init__(self, bundle=None):
        self.bundle = bundle

    def unit_z(self, rows):
        """학습된 기준 방식으로 창 z를 계산"""
        if self.bundle["baseline"] == "reference":
            return reference_z(rows, self.bundle["reference"])
        return per_unit_z(rows, self.bundle.get("feats", FEATS))

    def fit(self, normal, fault, baseline="per_unit", feats=FEATS):
        """normal: 정상 로봇들의 창(특징 dict), fault: 고장 라벨이 있는 로봇들의 창
        1) 정상 기준 2) 이상 감지 + 건강 점수 눈금 3) 원인 분류 순서로 학습"""
        # 1) 정상 로봇을 나눔: 모델 학습용(fit) / 눈금 보정용(cal). 보정을 학습에 안 쓴 로봇으로 해야
        #    '처음 보는 정상 로봇의 99.5%가 3점 아래'라는 눈금이 실제로 성립한다
        units = sorted({r["robot"] for r in normal})
        fit_units, cal_units = split_units(units)
        fit_rows = [r for r in normal if r["robot"] in fit_units]   # 모델 학습에 쓸 정상 창
        cal_rows = [r for r in normal if r["robot"] in cal_units]   # 눈금(3점 기준) 보정에 쓸 정상 창
        ref = None                                         # 공통 기준 (reference 방식일 때만 만듦)
        if baseline == "reference":
            ref = SegmentBaseline().fit(usable(fit_rows, feats), feats)          # 보정용 로봇은 기준에 넣지 않아야 '처음 보는 로봇' 눈금이 됨
            _, Zfit = reference_z(fit_rows, ref)
            _, Zcal = reference_z(cal_rows, ref)
            frows, Zf = reference_z(fault, ref)
        else:
            _, Zfit = per_unit_z(fit_rows, feats)
            _, Zcal = per_unit_z(cal_rows, feats)
            frows, Zf = per_unit_z(fault, feats)
        if not len(Zcal):                                  # 정상 로봇이 1대뿐이면 그 로봇의 뒤쪽 30%를 보정에 씀
            cut = int(len(Zfit) * 0.7)
            Zfit, Zcal = Zfit[:cut], Zfit[cut:]

        # 2) 이상 감지 5종을 정상 z로 학습하고, 건강 점수 눈금을 맞춘다 (3점 = 정상 99.5%, 9점 = 고장 기준 창의 중앙값)
        sev = np.array([r["true_sev"] for r in frows], float)        # 고장 로봇 창마다 라벨의 고장 정도
        Zfail, scale = Zf[(sev >= C.FAIL_CAL_BAND[0]) & (sev <= C.FAIL_CAL_BAND[1])], "band"   # 고장 정도 0.75~0.85 창
        if not len(Zfail):                                 # 고장 정도가 없거나 그 구간에 창이 없으면 고장 창 전체로 맞춤
            Zfail = Zf[sev >= C.CAUSE_MIN_SEVERITY]
            scale = "all_fault" if len(Zfail) else "normal_only"
        dets = make_detectors()
        for d in dets.values():
            calibrate(d.fit(Zfit), Zcal, Zfail)            # 학습은 fit 로봇, 3점 눈금은 cal 로봇, 9점 눈금은 고장 창
        main = "ae"                                         # 진단에 쓸 주 모델
        wheel_idx = [i for i, f in enumerate(feats) if f in WHEEL_FEATS]
        if wheel_idx and C.WHEEL_SCORE:                    # 엔코더가 있으면: 바퀴 하나에만 생긴 이상을 따로 보는 점수를 더함
            # 오토인코더는 특징 47개의 복원 오차를 평균 내므로 바퀴 하나의 이상이 묻히기 쉽다 → 바퀴 특징 중 가장 크게
            # 벗어난 값을 따로 보고, 두 점수를 같은 눈금(정상 99.5% = 3점)으로 맞춘 뒤 큰 쪽을 건강 점수로 쓴다
            dets["wheel"] = calibrate(WheelMaxDetector(feats).fit(Zfit), Zcal, Zfail)
            dets["ae_wheel"] = calibrate(CombinedDetector(dets["ae"], dets["wheel"]).fit(Zfit), Zcal, Zfail)
            main = "ae_wheel"

        # 3) 원인 분류: 고장 창(라벨 = 종류|바퀴) + 보정용 정상 창(라벨 = 정상)으로 랜덤포레스트 학습
        cause = None
        if (sev > 0).any():                                # 고장 초반(CAUSE_MIN_SEVERITY 미만)은 패턴이 모호하므로 제외
            keep = (sev == 0) | (sev >= C.CAUSE_MIN_SEVERITY)   # 고장 전(정상) 창 + 고장이 충분히 진행된 창
            # 창마다 정답 라벨: 정상이면 'normal|', 고장이면 '종류|바퀴' (예: 'bearing|fr')
            labels = np.array([NORMAL_LABEL if r["true_sev"] == 0 else f"{r['true_kind']}|{r['true_side']}" for r in frows])
            Zc = np.vstack([Zf[keep], Zcal])               # 입력: 고장 로봇 창 + 보정용 정상 로봇 창
            yc = np.concatenate([labels[keep], [NORMAL_LABEL] * len(Zcal)])
            if len(set(yc)) >= 2:                          # 라벨이 두 종류 이상이어야 분류를 배울 수 있음
                cause = CauseClassifier().fit(Zc, yc, feats)

        self.bundle = dict(detectors=dets, cause=cause, feats=list(feats), version="real-vib-1.0", baseline=baseline,
                           reference=ref, scale=scale, main=main, kalman=dict(C.KALMAN), fit_units=fit_units, cal_units=cal_units)
        return self

    def save(self, path=C.BUNDLE_PATH):
        """bundle 전체(모델 5종 + 원인 모델 + 기준 + 설정)를 파일 하나로 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)
        return path

    @classmethod
    def load(cls, path=C.BUNDLE_PATH):
        """저장한 파일을 읽어 파이프라인을 되살림 (diagnose 명령이 씀)"""
        return cls(joblib.load(Path(path)))

    def monitor(self, detector=None, use_cause=True):
        """로봇 1대를 실시간으로 지켜보는 추론기"""
        return Predictor(self.bundle, detector, use_cause)
