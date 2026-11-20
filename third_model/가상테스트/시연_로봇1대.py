"""[역할] 로봇 1대 시뮬레이션 화면 만들기 (11/20)

    python 가상테스트/시연_로봇1대.py [--kind unbalance --wheel fr --final 0.8 --ramp 6 --minutes 10]

가상 로봇 1대를 만들어(실측보정_로봇.py) 학습된 AI 모델(가상테스트_결과/성능시험/model_bundle.joblib)에
1초씩 실제로 넣고, 1초마다의 결과(건강 점수 · 상태 · 원인 · 의심 바퀴 · 근거 · 센서 값)를 기록해
화면 파일 가상테스트_결과/시연/로봇1대_시뮬레이션.html 을 만든다 (브라우저로 열면 재생).
  · 처음 5분은 AI가 이 로봇의 '평소'를 배우는 시간 → 화면은 그 뒤(0초)부터 재생
  · 0~5초: 정상 주행 → 5초부터 약한 고장이 생겨 조금 커진 뒤 약하게 이어짐 (기본: 앞 왼쪽 바퀴 불균형 0.12 → 3분 동안 0.24)
  · 기본값에서 AI는 3분에 '의심 중'(1단계 경보)을 내고 끝까지 '경고'까지는 가지 않음 (--final을 키우면 경고까지 감)
  · 화면의 숫자는 지어낸 값이 아니라 AI 모델이 실제로 낸 값. 단 로봇은 가상 로봇(실측 진동 조각 + 가정한 고장)
화면 파일에는 실측 진동 조각이 들어 있으므로 GitHub에 올리지 않는다 (가상테스트_결과/는 .gitignore)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))
sys.path.append(str(HERE))
import config as C  # noqa: E402
from pipeline import dataio as D  # noqa: E402
from pipeline.models import NORMAL_LABEL, label_ko  # noqa: E402
from pipeline.system import AMRPipeline  # noqa: E402
import sim_robots as S  # noqa: E402
import 실측보정_로봇 as R  # noqa: E402

KIND_SHOW = {"unbalance": "불균형", "looseness": "체결 풀림", "bearing": "마찰 증가 (베어링)", "unknown": "처음 보는 이상"}
TODO = {"unbalance": "바퀴 고정 상태 · 변형 확인, 바퀴에 붙은 이물질 제거",
        "looseness": "모터 브래킷 · 바퀴 나사 조이기",
        "bearing": "바퀴 축 · 베어링 걸림 확인, 윤활",
        "unknown": "해당 바퀴 주변을 직접 점검 (학습에 없던 이상일 수 있음)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="unbalance", choices=["unbalance", "looseness", "bearing"])
    p.add_argument("--wheel", default="fl", choices=["fl", "fr", "rl", "rr"])
    p.add_argument("--start", type=float, default=5.0, help="평소 기준(5분)이 끝난 뒤 몇 초에 고장이 시작되나")
    p.add_argument("--first", type=float, default=0.12, help="고장이 처음 나타날 때의 정도 (약함)")
    p.add_argument("--final", type=float, default=0.24, help="고장이 커져서 도달하는 정도")
    p.add_argument("--ramp", type=float, default=3.0, help="first → final까지 걸리는 시간 [분]")
    p.add_argument("--minutes", type=float, default=10.0, help="평소 기준 뒤에 보여 줄 시간 [분]")
    p.add_argument("--seed", type=int, default=2031)
    p.add_argument("--model", default=str(PROJ / "가상테스트_결과" / "성능시험" / "model_bundle.joblib"))
    p.add_argument("--out", default=str(PROJ / "가상테스트_결과" / "시연" / "로봇1대_시뮬레이션.html"))
    a = p.parse_args()

    base_min = C.minutes(C.BASELINE_PERIODS)                       # 평소 기준 시간 [분] (5)
    pools = R.RealPools({"공3옥상": ["data/정상 데이터_공3옥상", "data/정상 데이터_공3옥상(11.16)"],
                         "욧골": ["data/정상 데이터_욧골(11.15)"]}, ["114650", "114717", "110211"])
    sp = pools.split(a.seed)
    idle = [x for _, x in pools.idle]
    fp = R.RealPools({"욧골": ["data/고장 데이터_욧골"]})
    fm = R.FaultModel(sp["train"][("욧골", 5)], [x for _, x in fp.files[("욧골", 5)]])

    rng = np.random.default_rng(a.seed + 77)
    spec = S.complex_spec(rng, "시연_로봇", "점진 고장", kind=a.kind, wheel=a.wheel)
    onset = base_min + a.start / 60
    # severity(): 시작 0.1 → final. 처음부터 '약한' 정도(first)로 보이게 시작값을 맞춤
    spec.update(onset=onset, ramp=a.ramp, final=a.final, shape=1.0, strength=1.0, minutes=base_min + a.minutes,
                floor="공3옥상", spd=5, speed=1.0, dscale=1.0, cat="점진 고장")
    spec.pop("second", None)
    df = R.make_table(spec, sp["test"], idle, fm).drop(columns=["gx", "gy", "gz"])
    if a.first > 0.1:                                               # 고장 정도 열을 '시작 first'로 다시 계산해 다시 만듦
        S_sev = S.severity
        S.severity = lambda sp_, t: np.where(t >= sp_["onset"],
                                             a.first + (sp_["final"] - a.first) * np.clip((t - sp_["onset"]) / sp_["ramp"], 0, 1), 0.0)
        R.S.severity = S.severity
        df = R.make_table(spec, sp["test"], idle, fm).drop(columns=["gx", "gy", "gz"])
        S.severity = R.S.severity = S_sev
    recs = D._windows(df, spec["name"], D.Dataset(), None)[0]

    pipe = AMRPipeline.load(a.model)
    b = pipe.bundle
    mon = pipe.monitor()
    cls = b["cause"].classes
    t0 = base_min * 60
    frames, minutes = [], []
    last_status = 0
    for rec in recs:
        out = mon.update(rec)
        t = float(rec["t"]) - t0
        if out["phase"] == "learning" or t < -0.5:
            continue
        # 1초 원인 확률 → 바퀴별 의심 정도 (정상 · 처음 보는 이상 제외), 종류별 합
        wheel_p = {w: 0.0 for w in C.WHEEL_KO}
        kind_p = {}
        if "z" in out:
            pc = b["cause"].proba(out["z"][None, :])[0]
            for c_, v in zip(cls, pc):
                if c_ == NORMAL_LABEL:
                    continue
                k, w = c_.split("|")
                kind_p[k or "unknown"] = kind_p.get(k or "unknown", 0.0) + float(v)
                if w in wheel_p:
                    wheel_p[w] += float(v)
            normal_p = float(pc[cls.index(NORMAL_LABEL)]) if NORMAL_LABEL in cls else 0.0
        else:
            normal_p = 1.0
        x = {k: np.asarray(rec[k], float) for k in ("ax", "ay", "az") if k in rec}
        idx = np.linspace(0, len(x["az"]) - 1, 48).astype(int)
        cause_label = out.get("cause_label") or ""
        ck, cw = (cause_label.split("|") + [""])[:2] if cause_label and cause_label != NORMAL_LABEL else ("", "")
        cause_txt = out.get("cause") or ""
        unknown = bool(cause_txt) and "미확정" in cause_txt
        frames.append(dict(
            t=round(t, 1), sev=round(float(rec["true_sev"]), 3),
            h=round(float(out.get("window_score", 0) or 0), 2), ew=round(float(out.get("score") or 0), 2),
            st=int(out["status"]), cause=cause_txt, ck=("unknown" if unknown else ck), cw=("" if unknown else cw),
            reason=[[n, round(float(v), 2)] for n, v in out.get("cause_reason", [])][:3],
            top=[[n, round(v, 1)] for n, v in out.get("top_feats", [])][:3],
            wp={w: round(v, 3) for w, v in wheel_p.items()}, kp={k: round(v, 3) for k, v in kind_p.items()}, np=round(normal_p, 3),
            spd={w[-2:]: round(abs(float(rec.get(w, np.nan))), 0) if np.isfinite(rec.get(w, np.nan)) else None for w in C.ENCODER_CHANNELS},
            cur=round(float(rec.get("current", np.nan)), 3), volt=round(float(rec.get("voltage", np.nan)), 3),
            cmd=round(float(rec.get("motor_cmd", np.nan)), 1) if np.isfinite(rec.get("motor_cmd", np.nan)) else None,
            rms=round(float(np.sqrt(np.mean((x["az"] - x["az"].mean()) ** 2))), 4),
            wav=[[round(float(v - x[k].mean()), 3) for v in x[k][idx]] for k in ("ax", "az")]))
        if out["status"] != last_status:
            last_status = out["status"]
    mon.finish()
    for per, s in mon.period_scores:
        if per >= C.BASELINE_PERIODS:
            minutes.append(dict(m=round(C.minutes(per + 1) - base_min, 2), s=round(float(s), 2)))

    first_alarm = next((f["t"] for f in frames if f["st"] >= 1), None)
    meta = dict(kind=a.kind, kind_ko=KIND_SHOW[a.kind], wheel=a.wheel, wheel_ko=C.WHEEL_KO[a.wheel], start=a.start, first=a.first,
                final=a.final, ramp=a.ramp, caution=getattr(C, "CAUTION_ALARM", 2.0), warning=C.THRESH["warning"],
                confirm=int(b.get("confirm_periods", 2)), main=b.get("main"), first_alarm=first_alarm,
                todo=TODO, kind_show=KIND_SHOW, wheel_show=C.WHEEL_KO, minutes=minutes)
    tpl = (HERE / "시연_화면_틀.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(dict(meta=meta, frames=frames), ensure_ascii=False, separators=(",", ":")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    st = [f["st"] for f in frames]
    print(f"저장: {out} ({len(frames)}초, {out.stat().st_size / 1e6:.1f} MB)")
    print(f"첫 경보: {first_alarm}초 · 상태 변화: " + " → ".join(f"{t}초 {C.STATUS_KO[s]}" for t, s in
                                                           [(f['t'], f['st']) for i, f in enumerate(frames) if i == 0 or f['st'] != st[i - 1]]))
    last = frames[-1]
    print(f"마지막 원인: {last['cause']} · 근거 {last['reason']}")
    print("1분 점수:", [(m["m"], m["s"]) for m in minutes])


if __name__ == "__main__":
    main()
