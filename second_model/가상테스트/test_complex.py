"""[역할] 복잡한 고장까지 넣은 가상 로봇으로 충분히 학습하고, 학습에 없던 어려운 로봇까지 넣어 시험한다 (코드 점검용, 성능 아님)

    python 가상테스트/test_complex.py                 # 학습 160대 → 시험 170대 (40분 안팎)
    python 가상테스트/test_complex.py --scale 0.3     # 대수를 30%로 줄여 빠르게 (작동 확인용)

학습 로봇 160대 (복잡한 고장도 배우게)
    정상 50대   = 정상 30 · 바닥 바뀜 10 · 짐 실음 6 · 저전압 정상 4   ← '고장이 아닌 변화'도 정상으로 배움
    고장 110대  = 점진 고장 48 (종류 3 × 바퀴 4 × 4대) · 급성 12 · 간헐 12 · 복합 16 · 약한 12 · 속도 다름 10
시험 로봇 170대 (학습에 전혀 쓰지 않음)
    일반 100대       = 지난 '로봇100대' 시험과 같은 방식 (정상/고장 반반, 점진 고장)
    복잡한 고장 50대  = 급성 8 · 간헐 8 · 복합 8 · 약한 8 · 속도 극단 6 · 베어링 변형 6 · 고장+저전압 6
                       (속도 극단과 베어링 변형은 학습 때 없던 값: 처음 보는 종류의 어려움)
    어려운 정상 20대  = 바닥 바뀜 8 · 짐 실음 6 · 저전압 정상 6   ← 오경보 시험

'충분한 학습'의 뜻: 모델을 더 오래 돌리는 것이 아니라 더 많고 다양한 기록으로 배우게 하는 것이다.
오토인코더는 검증 오차가 더 줄지 않으면 스스로 멈추게(early stopping) 되어 있고, 억지로 더 돌리면 학습 기록을 외운다(과대적합).
칼만 필터 잡음 값도 학습 고장 로봇으로 맞춘다(실제 train과 같음). 모델 코드(pipeline/)는 바꾸지 않는다.

주의: 신호와 고장은 사람이 식으로 만든 것이라 이 %는 실제 성능이 아니다. 가상 데이터는 data/에 쓰지 않는다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from pipeline import dataio as D, report, workflow as W  # noqa: E402
from pipeline.models import NORMAL_LABEL, label_ko  # noqa: E402
import sim_robots as S  # noqa: E402

pct = lambda a, b: round(100.0 * a / b, 1) if b else None   # a/b → % (분모가 0이면 None)

TRAIN = {"정상": 30, "바닥 바뀜": 10, "짐 실음": 6, "저전압 정상": 4,
         "급성 고장": 12, "간헐 고장": 12, "복합 고장": 16, "약한 고장": 12, "속도 다름": 10}
TRAIN_GRADUAL_EACH = 4                                     # 점진 고장: 종류 × 바퀴 조합마다 몇 대
TEST_COMPLEX = {"급성 고장": 8, "간헐 고장": 8, "복합 고장": 8, "약한 고장": 8, "속도 다름": 6, "베어링 변형": 6, "고장+저전압": 6}
TEST_HARD_NORMAL = {"바닥 바뀜": 8, "짐 실음": 6, "저전압 정상": 6}
TEST_GENERAL = 100


def add_robot(data, spec):
    """가상 로봇 1대의 표 → 1초 창으로 잘라 Dataset에 넣음 (dataio.load가 CSV 한 파일을 읽을 때와 같은 함수)"""
    recs, fs = D._windows(S.make_table(spec), spec["name"], data, None)
    data.robots[spec["name"]], data.fs[spec["name"]] = recs, fs


def main():
    p = argparse.ArgumentParser(description="복잡한 고장 가상 로봇 학습·시험 (코드 점검용)")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--scale", type=float, default=1.0, help="로봇 대수 배율 (0.3이면 30%%로 줄여 빠르게)")
    p.add_argument("--train-scale", type=float, default=None, help="학습 로봇만 따로 배율 (없으면 --scale). 0.45 ≈ 학습 6분, 0.8 ≈ 10분")
    p.add_argument("--test-scale", type=float, default=None, help="시험 로봇만 따로 배율 (없으면 --scale). 0.5 ≈ 시험 6.5분, 0.75 ≈ 10분")
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과"))
    p.add_argument("--model-out", default=None, help="학습한 모델을 저장할 곳 (기본: 저장 안 함. artifacts/에는 쓰지 않음)")
    p.add_argument("--encoders", action="store_true", help="모든 가상 로봇에 엔코더 4개·모터 명령을 넣음 (바퀴 특징까지 학습)")
    p.add_argument("--glitches", action="store_true", help="모든 가상 로봇에 센서 읽기 오류(샘플 하나만 튀는 값, 1초에 0.2번꼴)를 넣음")
    p.add_argument("--set", action="append", default=[], metavar="이름=값",
                   help="config 값을 이번 실행에서만 바꿈 (예: --set PERIOD_QUANTILE=0.75 --set AE_ENSEMBLE=1)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f}분] {s}", flush=True)

    # 1) 로봇 설정 (학습 / 시험)
    train_specs, _ = make_specs(a.seed, a.train_scale or a.scale)   # 학습·시험 규모를 따로 정할 수 있음
    _, test_specs = make_specs(a.seed + 1 if a.train_scale or a.test_scale else a.seed, a.test_scale or a.scale)
    for s in train_specs + test_specs:                    # 같은 로봇들에 엔코더·읽기 오류만 더함 (진동 신호는 그대로)
        s["encoders"] = a.encoders
        s["glitch"] = 0.2 if a.glitches else 0
    for kv in a.set:                                      # 설정 바꾸기 (숫자면 숫자로, 아니면 글자로)
        k, v = kv.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            v = {"True": True, "False": False}.get(v, v)
        setattr(C, k, v)
    a.set_note = ", ".join(a.set)
    log(f"학습 로봇 {len(train_specs)}대 · 시험 로봇 {len(test_specs)}대")
    from pipeline.heartbeat import Heartbeat
    with Heartbeat("가상 데이터 만드는 중") as hb:        # 지금 하는 작업을 시작할 때 한 번, 그 뒤 5분마다 알림
        a.hb = hb
        run(a, out, train_specs, test_specs, t0, log)


def make_specs(seed, scale=1.0):
    """학습·시험 로봇 설정 목록 (같은 seed·scale이면 항상 같은 로봇들). compare_period.py도 이 함수로 같은 시험 로봇을 다시 만든다"""
    rng = np.random.default_rng(seed)
    nn = lambda v: max(1, int(round(v * scale)))
    train_specs = []
    for k in S.KINDS:
        for w in S.WHEELS:
            train_specs += [S.complex_spec(rng, f"학습_점진_{k}_{w}_{j + 1}", "점진 고장", k, w) for j in range(nn(TRAIN_GRADUAL_EACH))]
    for cat, cnt in TRAIN.items():
        train_specs += [S.complex_spec(rng, f"학습_{cat.replace(' ', '')}_{j + 1:02d}", cat) for j in range(nn(cnt))]
    test_specs = []
    for i in range(nn(TEST_GENERAL)):                       # 일반: 지난 100대 시험과 같은 방식
        s = S.random_spec(rng, f"일반{i + 1:03d}", rng.random() < 0.5)
        s.update(cat="일반 고장" if s["fault"] else "일반 정상", group="일반")
        test_specs.append(s)
    for cat, cnt in TEST_COMPLEX.items():
        for j in range(nn(cnt)):
            s = S.complex_spec(rng, f"복잡_{cat.replace(' ', '')}_{j + 1:02d}", cat, extreme=True)
            s["group"] = "복잡한 고장"
            test_specs.append(s)
    for cat, cnt in TEST_HARD_NORMAL.items():
        for j in range(nn(cnt)):
            s = S.complex_spec(rng, f"어려운정상_{cat.replace(' ', '')}_{j + 1:02d}", cat, extreme=True)
            s["group"] = "어려운 정상"
            test_specs.append(s)
    return train_specs, test_specs


def run(a, out, train_specs, test_specs, t0, log):
    """가상 기록 만들기 → 학습 → 시험 → 결과 저장"""
    # 2) 가상 기록 만들기 (메모리에서만)
    data = D.Dataset()
    allspecs = train_specs + test_specs
    for spec in allspecs:
        add_robot(data, spec)
    gen_s = time.time() - t0
    log(f"가상 기록 {len(allspecs)}대 만듦 ({gen_s / 60:.1f}분)")
    test_names = [s["name"] for s in test_specs]
    train_min = sum(s["minutes"] for s in train_specs)

    # 3) 학습 → 시험. 채점은 실측 교차검증과 같은 코드(cross_validate)를 '겹 1개 = 시험 로봇 전체'로 불러서 함
    #    실제 train처럼 칼만 필터 잡음도 학습 고장 로봇으로 맞추도록 tune=True로 학습
    orig_train, box = W.train, {}

    def train_full(data_, units=None, baseline=None, tune=True):
        a.hb.set(f"학습 중 (로봇 {len(units)}대, 기록 {train_min / 60:.1f}시간 분량)")
        t = time.time()
        pipe, notes = orig_train(data_, units, baseline, tune=True)
        box.update(pipe=pipe, notes=notes, train_s=time.time() - t, test_t0=time.time())
        log(f"학습 끝 ({(time.time() - t) / 60:.1f}분)")
        a.hb.set(f"평가 중 (처음 보는 시험 로봇 {len(test_names)}대 진단·채점)")
        return pipe, notes

    W.train = train_full                                 # 진행 알림은 Heartbeat가 5분마다 (pipeline/heartbeat.py)
    W.make_folds = lambda data_, names: [test_names]
    m = W.cross_validate(data, "per_unit", progress=lambda s: None)
    test_s = time.time() - box["test_t0"]              # 시험(진단·채점)에만 걸린 시간
    log(f"시험 끝 ({test_s / 60:.1f}분), 결과 정리")
    if a.model_out:
        box["pipe"].save(Path(a.model_out))

    # 4) 로봇별 결과 표
    classes = box["pipe"].bundle["cause"].classes if box["pipe"].bundle["cause"] is not None else []
    ko2raw = {label_ko(c): c for c in classes}
    ko2raw[label_ko(NORMAL_LABEL)] = NORMAL_LABEL
    by_name = {u["robot"]: u for u in m["unit_results"]}
    rows = []
    for s in test_specs:
        u = by_name.get(s["name"])
        if u is None:
            continue
        pred = ko2raw.get(u["cause_pred"]) if u["cause_pred"] else None
        kinds = {s["kind"]} | ({s["second"]["kind"]} if s.get("second") else set())   # 복합 고장은 두 종류 중 하나면 맞음
        alarm = u["caution"] is not None
        onset = s.get("onset")
        rows.append({
            "로봇": s["name"], "묶음": s["group"], "경우": s["cat"], "실제": "고장" if s["fault"] else "정상",
            "고장 종류": "+".join(C.KIND_KO[k] for k in [s["kind"]] + ([s["second"]["kind"]] if s.get("second") else [])) if s["fault"] else "",
            "고장 바퀴": C.WHEEL_KO.get(s["wheel"], "") if s["fault"] else "",
            "최종 고장 정도": round(s["final"], 2) if s["fault"] else 0.0, "고장 시작(분)": round(onset, 1) if s["fault"] else None,
            "기록(분)": round(s["minutes"], 1), "최고 점수": u["max_score"],
            "첫 주의(분)": C.minutes(u["caution"] + 1) if alarm else None,
            "첫 경고(분)": C.minutes(u["warning"] + 1) if u["warning"] is not None else None,
            "감지까지(분)": round(C.minutes(u["caution"] + 1) - onset, 1) if (alarm and s["fault"]) else None,
            "판정": ("맞음 (고장 감지)" if alarm else "놓침") if s["fault"] else ("오경보" if alarm else "맞음 (정상)"),
            "추정 원인": u["cause_pred"] or "",
            "종류 맞음": (pred.split("|")[0] in kinds) if (s["fault"] and pred and pred != NORMAL_LABEL) else None,
            "경고 여유(분)": C.minutes(u["lead_periods"]) if u["lead_periods"] is not None and u["lead_periods"] >= 0 else None,
            "정지 위험까지 감": u["fail"] is not None})
    df = pd.DataFrame(rows)
    df.to_csv(out / "로봇별_결과.csv", index=False, encoding="utf-8-sig")

    def stats(d):
        """로봇 묶음 하나 → % 요약"""
        F, N = d[d["실제"] == "고장"], d[d["실제"] == "정상"]
        hit, ok = int(F["첫 주의(분)"].notna().sum()), int(N["첫 주의(분)"].isna().sum())
        fl = F[F["정지 위험까지 감"]]
        warned = int(fl["경고 여유(분)"].notna().sum())
        cz = F[F["종류 맞음"].notna()]
        return dict(robots=len(d), fault=len(F), normal=len(N), accuracy_pct=pct(hit + ok, len(d)),
                    detect_pct=pct(hit, len(F)), false_alarm_pct=pct(len(N) - ok, len(N)),
                    warned_before_fail_pct=pct(warned, len(fl)), fail_robots=len(fl),
                    mean_detect_delay_min=round(float(F["감지까지(분)"].mean()), 1) if hit else None,
                    kind_ok_pct=pct(int(cz["종류 맞음"].sum()), len(cz)), cause_robots=len(cz))

    res = dict(all=stats(df), groups={g: stats(df[df["묶음"] == g]) for g in ["일반", "복잡한 고장", "어려운 정상"]},
               cases={c: stats(df[df["경우"] == c]) for c in dict.fromkeys(df["경우"])},
               windows={v["name"]: dict(auc=v["auc"], detect_pct=None if v["recall"] is None else round(100 * v["recall"], 1),
                                        false_alarm_pct=None if v["false_positive"] is None else round(100 * v["false_positive"], 1))
                        for v in m["detectors"].values()},
               forecast=m["forecast"], train=dict(robots=len(train_specs), hours=round(train_min / 60, 1),
                                                   minutes=round(box["train_s"] / 60, 1), notes=box["notes"]),
               test_minutes=round(test_s / 60, 1), seed=a.seed, scale=a.scale, skipped=m["skipped"])
    (out / "metrics.json").write_text(json.dumps(dict(summary=res, cv_metrics=m), ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    report.fig_auc(m, out / "fig_auc.png")
    report.fig_confusion(m, out / "fig_confusion.png")
    write_md(res, df, out / "README.md", a, train_specs)
    print(json.dumps({k: res[k] for k in ("all", "groups", "train", "test_minutes")}, ensure_ascii=False, indent=1))
    log(f"저장: {out}")


def write_md(r, df, path, a, train_specs):
    """결과 % → README.md"""
    f = lambda v, unit="%": "-" if v is None else f"{v:g}{unit}"
    tr = r["train"]
    k = r["forecast"]["kalman"]
    cnt = pd.Series([s["cat"] for s in train_specs]).value_counts()
    lines = [
        "# 복잡한 고장 가상 로봇 학습·시험 결과 (코드 점검용 · 성능 아님)", "",
        "> **이 %는 발표·성능 자료로 쓰지 않습니다.** 진동과 고장을 사람이 식으로 만든 가상 로봇이라 실제 로봇 성능이 아닙니다.",
        "> 이 시험이 보여 주는 것: 어떤 종류의 어려운 경우에서 이 AI 구조가 약한지 (실측 기록을 계획할 때 참고).", "",
        f"- 실행: `가상테스트/test_complex.py --seed {a.seed}" + (f" --scale {a.scale:g}" if a.scale != 1 else "") + (f" --train-scale {a.train_scale:g}" if a.train_scale else "") + (f" --test-scale {a.test_scale:g}" if a.test_scale else "") + (" --encoders" if a.encoders else "") + (" --glitches" if a.glitches else "") + "".join(f" --set {x}" for x in a.set) + "` (같은 명령이면 같은 결과)",
        f"- 학습: 가상 로봇 {tr['robots']}대, 기록 {tr['hours']}시간 분량, 학습에 걸린 시간 {tr['minutes']}분 (칼만 필터 조정 포함)",
        "  - " + " · ".join(f"{c} {n}" for c, n in cnt.items()),
        f"- 시험: 학습에 쓰지 않은 가상 로봇 {r['all']['robots']}대, 걸린 시간 {r['test_minutes']}분",
        "  - 일반: 지난 `로봇100대` 시험과 같은 방식 (정상/고장 반반, 점진 고장)",
        "  - 복잡한 고장: 급성 · 간헐 · 복합(두 고장) · 약한 고장 · 속도 극단(학습에 없던 속도) · 베어링 변형(학습에 없던 주파수 120/320 Hz) · 고장+저전압",
        "  - 어려운 정상: 기록 중간에 바닥이 거칠어짐 · 중간에 짐을 실음 · 배터리 저전압 (고장이 아니므로 경보가 뜨면 오경보)",
        f"- 설정: 기본값 그대로 (판정 주기 {C.PERIOD_S:g}초 · 정상 기준 per_unit · 주의 3점 / 경고 6점 / 위험 9점)",
        f"- 엔코더: {'4개 + 모터 명령 (바퀴 특징 11개 추가)' if a.encoders else '없음 (진동 36개만)'}"
        + (" · 센서 읽기 오류(튀는 값) 넣음" if a.glitches else ""),
        f"- 바꾼 설정: {a.set_note or '없음'} · 1분 점수 지점 {C.PERIOD_QUANTILE:g} · 오토인코더 {C.AE_ENSEMBLE}개 · 보정 로봇 {C.CAL_SPLIT} · 튀는 값 기준 {C.SPIKE_K:g}", "",
        "## 묶음별 결과", "",
        "| 묶음 | 로봇 | 판정 정답률 | 고장 감지율 | 오경보율 | 정지 위험 전 경고 | 고장 시작 → 주의 | 고장 종류 정확도 |",
        "|---|---|---|---|---|---|---|---|"]
    for g, v in [("**전체**", r["all"])] + list(r["groups"].items()):
        lines.append(f"| {g} | {v['robots']}대 | **{f(v['accuracy_pct'])}** | {f(v['detect_pct'])} | {f(v['false_alarm_pct'])} | "
                     f"{f(v['warned_before_fail_pct'])} ({v['fail_robots']}대 중) | {f(v['mean_detect_delay_min'], '분')} | "
                     f"{f(v['kind_ok_pct'])} ({v['cause_robots']}대 중) |")
    lines += ["", "## 경우별 결과", "",
              "| 경우 | 로봇 | 판정 정답률 | 고장 감지율 | 오경보율 | 정지 위험 전 경고 | 고장 시작 → 주의 | 고장 종류 정확도 |",
              "|---|---|---|---|---|---|---|---|"]
    for c, v in r["cases"].items():
        lines.append(f"| {c} | {v['robots']}대 | {f(v['accuracy_pct'])} | {f(v['detect_pct'])} | {f(v['false_alarm_pct'])} | "
                     f"{f(v['warned_before_fail_pct'])} ({v['fail_robots']}대 중) | {f(v['mean_detect_delay_min'], '분')} | "
                     f"{f(v['kind_ok_pct'])} ({v['cause_robots']}대 중) |")
    lines += ["", "- 판정 정답률: 고장 로봇은 '주의' 이상이 뜨면, 정상 로봇은 경보가 없으면 맞음",
              "- 고장 시작 → 주의: 라벨상 고장이 시작된 뒤 '주의'가 뜰 때까지 걸린 평균 시간",
              "- 고장 종류 정확도: 원인을 낸 고장 로봇 중 종류를 맞힌 비율 (복합 고장은 두 종류 중 하나면 맞음)", "",
              "## 1초 창 단위 (모델 5개 비교, 시험 로봇 전체)", "",
              "| 이상 감지 방법 | AUC | 고장 창 감지율 (3점 이상) | 정상 창 오경보율 |", "|---|---|---|---|"]
    lines += [f"| {name} | {v['auc']} | {f(v['detect_pct'])} | {f(v['false_alarm_pct'])} |" for name, v in r["windows"].items()]
    lines += ["", f"남은 시간 예측(칼만 필터): 평균 오차 {f(k['mae_min'], '분')} ({k['n']}회) · 90% 범위 적중 "
              f"{f(None if k['coverage'] is None else round(100 * k['coverage']))}", ""]
    miss = df[df["판정"].isin(["놓침", "오경보"])]
    lines += ["## 틀린 로봇", ""]
    if len(miss):
        lines += ["| 로봇 | 경우 | 판정 | 실제 | 최종 고장 정도 | 최고 점수 |", "|---|---|---|---|---|---|"]
        lines += [f"| {x['로봇']} | {x['경우']} | {x['판정']} | {x['실제']} {x['고장 바퀴']} {x['고장 종류']} | {x['최종 고장 정도']} | {x['최고 점수']} |"
                  for _, x in miss.iterrows()]
    else:
        lines.append("없음")
    lines += ["", "## 학습 메모", ""] + [f"- {n}" for n in tr["notes"]]
    if r["skipped"]:
        lines += ["", "평가에서 빠진 로봇: " + "; ".join(r["skipped"])]
    lines += ["", f"파일: `로봇별_결과.csv` ({r['all']['robots']}대 전부), `metrics.json` (모든 수치), `fig_auc.png`, `fig_confusion.png`", "",
              "## 읽는 법", "",
              "- 일반 묶음은 지난 `로봇100대` 시험(학습 36대)과 같은 방식이라, 학습 로봇을 늘린 효과를 대략 비교할 수 있습니다 (시험 로봇은 다름).",
              "- 복잡한 고장 · 어려운 정상 묶음의 낮은 수치는 그 경우에 이 구조가 약하다는 신호입니다. 실측 기록 계획에 반영하세요",
              "  (예: 바닥이 바뀌는 곳은 SEGMENT로 구분해 기록, 간헐 고장은 오래 기록).",
              "- 가상 신호는 사람이 정한 식이므로, 실제 로봇에서 어떤 경우가 어려운지는 실측으로 다시 확인해야 합니다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
