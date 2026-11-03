"""AMR 고장예지 대시보드: 저장된 파이프라인으로 로봇 1대를 실시간처럼 재생한다.

    python run_all.py                  # 모델이 없을 때 먼저 1회
    streamlit run dashboard/app.py     # amr_pdm 폴더에서 실행 → http://localhost:8501

시연 흐름: 기준 학습 → 정상 → 주의 → 경고·원인·예측 → [정비 완료] → 정상 복귀
입력 데이터는 시뮬레이션과 내 데이터(CSV, 형식은 DATA_FORMAT.md) 중에서 고른다.
"""
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from pipeline import userdata as U
from pipeline.features import FEATS
from pipeline.models import NORMAL_LABEL
from pipeline.system import BUNDLE_PATH, AMRPipeline
from sim.simulator import AMRSimulator

# ---------------- 시연 설정 ----------------
START_LAP, N_LAPS, SEED = 8, 70, 600      # SEED: 학습·평가에 쓰지 않은 로봇, N_LAPS: 베어링 정비 후 복귀까지 여유 포함
TICK_S = float(os.environ.get("AMR_TICK_S", 1.0))   # 화면 갱신 주기: 그리기가 끝나기 전에 다음 갱신이 오면 아래쪽이 멈춤
BUDGET_S = TICK_S * 0.5                   # 한 번에 추론에 쓰는 최대 시간
HISTORY = 300                             # 센서 그래프에 보여 줄 최근 창 수
KINDS = {"베어링 결함": "bearing", "체결부 풀림": "looseness", "불평형": "unbalance", "정상 (고장 없음)": "normal"}
WHEELS = {"오른쪽 바퀴": "right", "왼쪽 바퀴": "left"}
RAMPS = {"보통 (24회차)": 24, "빠름 (16회차)": 16, "느림 (32회차)": 32}
SPEEDS = {"1회차/초": 65, "0.5회차/초": 33, "2회차/초": 130}    # 초당 처리할 창 수
ACTIONS = {"bearing": "베어링 소음을 확인하고 베어링 교체를 준비하세요.",
           "looseness": "바퀴와 모터의 체결 볼트 조임 상태를 확인하세요.",
           "unbalance": "바퀴의 이물질과 마모 상태를 확인하고 균형을 점검하세요."}
SIDE_PLAIN = {"left": "왼쪽", "right": "오른쪽"}

# ---------------- 색 (역할별) ----------------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LEFT, RIGHT, SERIES = "#2a78d6", "#eb6834", "#2a78d6"     # 왼쪽·오른쪽 바퀴는 어떤 그래프에서든 같은 색
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
FONT = '"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif'
CHART_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="AMR 고장예지", page_icon="🤖", layout="wide")
st.markdown(f"""<style>
.block-container {{padding-top:1.6rem; padding-bottom:2rem; max-width:1500px}}
.tile {{background:{SURFACE}; border:1px solid {GRID}; border-radius:10px; padding:12px 14px; height:92px; overflow:hidden}}
.tile .k {{font-size:13px; color:{INK2}}}
.tile .v {{font-size:24px; font-weight:600; color:{INK}; line-height:1.3}}
.tile .v.small {{font-size:17px; line-height:1.35}}
.tile .s {{font-size:12px; color:{INK2}}}
.pill {{display:inline-block; font-size:20px; font-weight:700; padding:5px 14px; border-radius:999px; margin-top:4px}}
.alert {{background:{SURFACE}; border:1px solid {GRID}; border-left:6px solid; border-radius:10px; padding:12px 16px; min-height:76px}}
.alert .t {{font-size:18px; font-weight:700; color:{INK}}}
.alert .b {{font-size:14px; color:{INK2}; margin-top:2px}}
.reason {{font-size:14px; color:{INK}; padding:3px 0; border-top:1px solid {GRID}}}
.head {{font-size:14px; font-weight:600; color:{INK2}; margin:0 0 2px}}
[data-stale="true"] {{opacity:1 !important}}
[data-testid="stToolbar"], [data-testid="stStatusWidget"] {{visibility:hidden}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(max_entries=2, show_spinner="모델 불러오는 중…")
def _load_pipeline(path, mtime):
    pipe = AMRPipeline.load(path)
    cause = pipe.bundle["cause"]
    if cause is not None:
        cause.explain(np.zeros(len(FEATS)), cause.classes[0]) # SHAP을 미리 준비해 시연 도중 멈추지 않게 함
    return pipe


def load_pipeline(path=BUNDLE_PATH):
    return _load_pipeline(path, path.stat().st_mtime)     # 파일이 바뀌면(다시 학습하면) 새로 읽음


@st.cache_resource(max_entries=1, show_spinner="데이터 읽는 중…")
def load_user_data(files, fs, lap_s):
    return U.load([f for f, _, _ in files], fs, lap_s)


class Run:
    """로봇 1대 재생 상태: 시뮬레이터 또는 내 데이터 → 파이프라인 → 화면용 기록"""

    def __init__(self, pipe, windows, n_laps, sim=None, kind=None, wheel=None):
        self.sim, self.kind, self.wheel, self.n_laps = sim, kind, wheel, n_laps
        self.windows, self.monitor = iter(windows), pipe.monitor()
        self.has_cause = pipe.bundle["cause"] is not None
        self.hist, self.log = deque(maxlen=HISTORY), [(0, "info", "주행 시작: 이 로봇의 정상 기준을 학습합니다")]
        self.rec, self.out, self.row = None, None, None
        self.phase, self.segment, self.lap, self.done, self.repaired = "learning", "-", 0, False, None
        self.seen_events = 0

    def step(self, n_windows):
        t0, n = time.perf_counter(), 0
        limit = n_windows * 4 if self.phase == "learning" else n_windows       # 기준 학습 구간은 빠르게 넘김
        while n < limit and time.perf_counter() - t0 < BUDGET_S:
            rec = next(self.windows, None)
            if rec is None:
                self.monitor.finish()
                self.done = True
                self.log.append((self.lap, "info", "시나리오 종료"))
                break
            o = self.monitor.update(rec)
            self.rec, self.row, self.lap, self.segment = rec, o["row"], o["lap"], o["segment"]
            if o["phase"] == "monitor":
                if self.phase == "learning":
                    self.log.append((self.lap, "info", "정상 기준 학습 완료: 실시간 진단 시작"))
                self.phase, self.out = "monitor", o
            self.hist.append({k: o["row"][k] for k in ("left_cur", "right_cur", "left_temp", "right_temp", "res_left", "res_right")})
            n += 1
        for lap, s in self.monitor.events[self.seen_events:]:
            self.log.append((lap, ["ok", "warn", "bad"][s], self._event_text(s)))
        self.seen_events = len(self.monitor.events)
        if self.repaired is not None and self.monitor.status == 0:
            self.repaired = None

    def _event_text(self, s):
        if s == 0:
            return "정상 범위로 복귀했습니다"
        if self.repaired is not None:
            return f"정비 후 점수 하강: {C.STATUS_KO[s]} 단계"
        cause = self.out["cause"] if self.out and self.out.get("cause") else "원인 분석 중"
        return f"{C.STATUS_KO[s]} 단계 진입: {cause}"


def simulated_run():
    s = st.session_state
    kind, wheel = KINDS[s.kind], WHEELS[s.wheel]
    faults = [] if kind == "normal" else [dict(kind=kind, wheel=wheel, start_lap=START_LAP, end_lap=START_LAP + RAMPS[s.ramp])]
    sim = AMRSimulator(faults=faults, n_laps=N_LAPS, seed=SEED)
    return Run(load_pipeline(), sim.windows(), N_LAPS, sim=sim, kind=kind, wheel=wheel)


def clear_run():
    """다음 화면 갱신 때 새 재생을 만든다"""
    st.session_state.pop("run", None)
    st.session_state.playing = False


# ---------------- 화면 조각 ----------------
def status_view(run):
    """상태 라벨, 글자색, 바탕색 (색만으로 구분되지 않도록 기호와 글자를 함께 씀)"""
    if run.phase == "learning":
        return "◌ 기준 학습 중", INK2, "#f0efec"
    st_, scores = run.monitor.status, run.monitor.lap_scores
    if st_ == 2 and run.monitor.calibrated and scores and scores[-1][1] >= C.THRESH["danger"]:
        return "■ 정지 위험", "#ffffff", CRITICAL
    return [("✓ 정상", "#ffffff", GOOD), ("! 주의", INK, WARNING), ("▲ 경고", "#ffffff", SERIOUS)][st_]


def tile(col, key, value, sub="", small=False):
    col.markdown(f'<div class="tile"><div class="k">{key}</div><div class="v{" small" if small else ""}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def draw_strip(run):
    label, fg, bg = status_view(run)
    mon = run.monitor
    c = st.columns([1.15, 1, 1, 1.5, 1.15])
    c[0].markdown(f'<div class="tile"><div class="k">현재 상태</div><span class="pill" style="color:{fg};background:{bg}">{label}</span></div>',
                  unsafe_allow_html=True)
    tile(c[1], "건강 점수", "–" if mon.ewma is None else f"{mon.ewma:.1f}", "주의 3 · 경고 6 · 정지 위험 9" if mon.calibrated else "주의 3 · 경고 6")
    tile(c[2], "주행", f"{run.lap}회차", run.segment)
    cause = "정비 후 확인 중" if run.repaired is not None else (run.out or {}).get("cause") or "–"
    if not run.has_cause:
        tile(c[3], "추정 원인", "원인 모델 없음", "고장 라벨을 넣어 학습하면 나옵니다", small=True)
    else:
        tile(c[3], "추정 원인", cause, "랜덤포레스트 + SHAP", small=True)
    fc = None if run.repaired is not None else mon.forecast
    if not mon.calibrated:
        tile(c[4], "정지 위험까지", "–", "고장 데이터로 학습하면 나옵니다")
    elif run.repaired is None and run.phase == "monitor" and mon.lap_scores and mon.lap_scores[-1][1] >= C.THRESH["danger"]:
        tile(c[4], "정지 위험까지", "도달", "즉시 점검")
    elif fc:
        tile(c[4], "정지 위험까지", f"약 {fc['laps']:.0f}회차", f"빠르면 {fc['laps_lo']:.0f}회차 · 칼만 필터")
    else:
        tile(c[4], "정지 위험까지", "–", "칼만 필터")


def draw_alert(run):
    """비전문가도 바로 행동할 수 있는 문구"""
    mon, out = run.monitor, run.out or {}
    label = out.get("cause_label")
    if run.phase == "learning":
        done = min(1.0, run.lap / C.BASELINE_LAPS)
        color, title, body = MUTED, "정상 기준 학습 중", f"처음 {C.BASELINE_LAPS}회 주행으로 이 로봇만의 구간별 정상 기준을 만듭니다 ({done:.0%})"
    elif run.repaired is not None:
        color, title, body = SERIES, "정비 완료, 정상 복귀 확인 중", "고장 성분은 사라졌고 모터 온도가 내려가는 동안 지켜봅니다."
    elif mon.status == 0:
        color, title, body = GOOD, "이상 없음", "구간별 정상 기준 안에서 주행하고 있습니다."
    else:
        side = kind = None
        if label and label != NORMAL_LABEL:
            kind, side = label.split("|")
        where = f"{SIDE_PLAIN[side]} 바퀴" if side else "구동부"
        fc = mon.forecast
        when = f"약 {fc['laps']:.0f}회차 후 정지 위험 (빠르면 {fc['laps_lo']:.0f}회차). " if fc and fc["laps"] > 0 else ""
        color = SERIOUS if mon.status == 2 else WARNING
        title = f"{where} 점검 필요"
        if kind:
            advice = ACTIONS.get(kind, f"{C.KIND_KO.get(kind, kind)} 항목을 점검하세요.")
        else:
            advice = "원인을 분석하고 있습니다." if run.has_cause else "정상 범위를 벗어났습니다. 구동부를 점검하세요."
        body = when + advice
    if run.sim is None:                                  # 이미 기록된 데이터는 정비로 바꿀 수 없음
        st.markdown(f'<div class="alert" style="border-left-color:{color}"><div class="t">{title}</div><div class="b">{body}</div></div>',
                    unsafe_allow_html=True)
        return
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.markdown(f'<div class="alert" style="border-left-color:{color}"><div class="t">{title}</div><div class="b">{body}</div></div>',
                  unsafe_allow_html=True)
    can_fix = run.phase == "monitor" and mon.status >= 1 and run.repaired is None and not run.done
    if right.button("🔧 정비 완료", disabled=not can_fix, width="stretch", key="repair"):
        run.sim.repair()
        run.repaired = run.lap
        run.log.append((run.lap, "info", "정비 완료: 다음 회차부터 고장 제거"))


def base_fig(height, legend=True):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=44, r=12, t=26 if legend else 8, b=30), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK2), showlegend=legend,
                      legend=dict(orientation="h", x=0, y=1.14, bgcolor="rgba(0,0,0,0)"), hovermode="x unified")
    axis = dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickcolor=GRID, automargin=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def health_fig(run):
    mon = run.monitor
    laps = [l for l, _ in mon.lap_scores]
    scores = [s for _, s in mon.lap_scores]
    top = max([C.THRESH["danger"] + 1.5] + scores)
    fig = base_fig(330)
    fig.update_xaxes(range=[0, run.n_laps], title_text="회차", title_standoff=6)
    fig.update_yaxes(range=[0, top], title_text="건강 점수", title_standoff=4)
    lines = [("주의", C.THRESH["caution"], WARNING), ("경고", C.THRESH["warning"], SERIOUS)]
    if mon.calibrated:
        lines.append(("정지 위험", C.THRESH["danger"], CRITICAL))
    for name, v, color in lines:
        fig.add_hline(y=v, line=dict(color=color, width=1, dash="dash"), annotation_text=f"{name} {v:.0f}",
                      annotation_position="top left", annotation_font=dict(size=12, color=INK2))
    fig.add_trace(go.Scatter(x=laps, y=scores, mode="lines+markers", name="회차 점수", line=dict(color=SERIES, width=2),
                             marker=dict(size=7, color=SERIES, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y:.1f}"))
    fc = None if run.repaired is not None else mon.forecast
    if fc and laps and fc["laps"] > 0:
        x1 = laps[-1] + fc["laps"]
        fig.add_trace(go.Scatter(x=[laps[-1], x1], y=[fc["level"], C.THRESH["danger"]], mode="lines", name="칼만 필터 예측",
                                 line=dict(color=INK2, width=2, dash="dot"), hovertemplate="%{y:.1f}"))
        if x1 <= run.n_laps:
            fig.add_annotation(x=x1, y=C.THRESH["danger"], text=f"예상 {x1:.0f}회차", showarrow=False, yshift=14,
                               font=dict(size=12, color=INK))
    return fig


def sensor_fig(run, left_key, right_key, ytitle, height=210):
    H = list(run.hist)
    fig = base_fig(height)
    x = list(range(-len(H) + 1, 1))
    for name, key, color in (("왼쪽", left_key, LEFT), ("오른쪽", right_key, RIGHT)):
        fig.add_trace(go.Scatter(x=x, y=[h[key] for h in H], mode="lines", name=name, line=dict(color=color, width=1.6),
                                 hovertemplate="%{y:.3f}"))
    fig.update_xaxes(title_text="최근 시간 (초)", title_standoff=4)
    fig.update_yaxes(title_text=ytitle, title_standoff=4)
    return fig


def wave_fig(run, height=210):
    fig = base_fig(height)
    if run.rec is not None:
        fs = run.rec.get("vib_fs", C.SENSORS["vib_fs"])
        n = min(int(0.5 * fs), len(run.rec["left_vib"]), len(run.rec["right_vib"]))
        step = max(1, n // 1000)                         # 샘플링이 높은 데이터는 솎아서 그림
        x = (np.arange(0, n, step) / fs).tolist()        # 배열 그대로 넘기면 Streamlit의 plotly.js가 못 그릴 수 있어 목록으로 변환
        for name, key, color in (("왼쪽", "left_vib", LEFT), ("오른쪽", "right_vib", RIGHT)):
            fig.add_trace(go.Scatter(x=x, y=run.rec[key][:n:step].tolist(), mode="lines", name=name, line=dict(color=color, width=1.2),
                                     hovertemplate="%{y:.3f} g"))
    fig.update_xaxes(title_text="시간 (초)", title_standoff=4)
    fig.update_yaxes(title_text="진동 [g]", title_standoff=4)
    return fig


def cause_fig(probs):
    items = probs[:4][::-1]
    fig = base_fig(170, legend=False)
    fig.add_trace(go.Bar(x=[p for _, p in items], y=[n for n, _ in items], orientation="h", marker=dict(color=SERIES),
                         width=0.55, text=[f"{p:.0%}" for _, p in items], textposition="outside", cliponaxis=False,
                         textfont=dict(color=INK), hovertemplate="%{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=12, t=4, b=4), hovermode="closest")
    return fig


def draw_cause(run):
    out, mon = run.out or {}, run.monitor
    st.markdown('<p class="head">원인 확률</p>', unsafe_allow_html=True)
    if out.get("cause_probs"):
        st.plotly_chart(cause_fig(out["cause_probs"]), use_container_width=True, theme=None, config=CHART_CONFIG, key="cause")
    else:
        st.caption("주의 단계에 들어가면 원인을 추정합니다.")
    st.markdown('<p class="head">판단 근거 (SHAP)</p>', unsafe_allow_html=True)
    if mon.reason:
        st.markdown("".join(f'<div class="reason">{name}</div>' for name, _ in mon.reason), unsafe_allow_html=True)
    else:
        st.caption("원인이 정해지면 가장 크게 작용한 특징을 보여 줍니다.")


def draw_log(run):
    tag = {"ok": "✓ 정상", "warn": "! 상태", "bad": "▲ 경고", "info": "· 기록"}
    rows = [dict(회차=lap, 구분=tag[t], 내용=msg) for lap, t, msg in reversed(run.log[-30:])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=180)


# ---------------- 실시간 탭 ----------------
def live_view():
    run = st.session_state.run
    if st.session_state.playing and not run.done:
        run.step(int(SPEEDS[st.session_state.speed] * TICK_S))
    draw_strip(run)
    st.write("")
    draw_alert(run)

    main, side = st.columns([3, 1.1])
    with main:
        st.markdown('<p class="head">건강 점수 추이와 정지 위험 예측</p>', unsafe_allow_html=True)
        st.plotly_chart(health_fig(run), use_container_width=True, theme=None, config=CHART_CONFIG, key="health")
    with side:
        draw_cause(run)

    c = st.columns(4)
    charts = [("진동 파형 (최근 창)", lambda: wave_fig(run), "wave"),
              ("모터 전류", lambda: sensor_fig(run, "left_cur", "right_cur", "전류 [A]"), "cur"),
              ("모터 온도", lambda: sensor_fig(run, "left_temp", "right_temp", "온도 [°C]"), "temp"),
              ("센서 불일치 (실제 − 예상 전류)", lambda: sensor_fig(run, "res_left", "res_right", "잔차 [A]"), "res")]
    for col, (title, make, key) in zip(c, charts):
        col.markdown(f'<p class="head">{title}</p>', unsafe_allow_html=True)
        col.plotly_chart(make(), use_container_width=True, theme=None, config=CHART_CONFIG, key=key)

    a, b = st.columns([1.3, 1])
    with a:
        st.markdown('<p class="head">이벤트 기록</p>', unsafe_allow_html=True)
        draw_log(run)
    with b:
        st.markdown('<p class="head">회차별 점수 표</p>', unsafe_allow_html=True)
        table = pd.DataFrame([dict(회차=l, 점수=round(s, 2)) for l, s in reversed(run.monitor.lap_scores)], columns=["회차", "점수"])
        st.dataframe(table, hide_index=True, width="stretch", height=180)
        if run.sim is None:
            st.download_button("회차별 점수 CSV 받기", table.sort_values("회차").to_csv(index=False).encode("utf-8-sig"),
                               "lap_scores.csv", "text/csv", key="download", on_click="ignore")
    if run.row is not None and run.sim is not None:
        sev = run.row["true_sev"]
        truth = "고장 없음" if run.kind == "normal" and run.repaired is None else \
            f"{SIDE_PLAIN[run.wheel]} 바퀴 {C.KIND_KO.get(run.kind, '')}, 고장 정도 {sev:.2f}"
        st.caption(f"시뮬레이션 정답 (모델에는 주지 않음): {truth}")
    elif run.rec is not None and run.rec["labeled"]:
        kind, sev = C.KIND_KO.get(run.rec["true_kind"], run.rec["true_kind"]), run.rec["true_sev"]
        st.caption(f"파일의 라벨 (모델에는 주지 않음): {kind}" + (f", 고장 정도 {sev:.2f}" if sev > 0 else ""))
    if run.done and st.session_state.playing:
        st.session_state.playing = False
        st.rerun()


# ---------------- 성능 보고서 탭 ----------------
def report_view():
    path = C.REPORT_DIR / "metrics.json"
    if not path.exists():
        st.info("`python run_all.py`를 먼저 실행하면 성능 보고서가 만들어집니다.")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    st.warning("시뮬레이션 데이터 결과입니다. 실측 검증 전이므로 발표 수치로 그대로 쓰지 마세요.", icon="⚠️")
    d, fk, ft = m["data"]["test"], m["forecast_compare"]["kalman"], m["forecast_compare"]["trend"]
    ae = m["detector_comparison"]["ae"]
    c = st.columns(4)
    tile(c[0], "첫 주의 시점 고장 정도", ae["mean_sev_at_caution"], "오토인코더, 낮을수록 일찍 감지")
    tile(c[1], "오경보 로봇", ae["false_alarm_robots"], "정상 로봇 기준")
    tile(c[2], "원인 분류 (로봇 단위)", m["cause_robot_accuracy"], f"창 단위 {m['cause_window_accuracy']:.0%}")
    tile(c[3], "정지 위험 예측 오차", f"{fk['mae']}회차", f"추세선 {ft['mae']}회차 · 90% 범위 적중 {fk['coverage']:.0%}")
    st.caption(f"평가: 처음 보는 테스트 로봇 {d['robots']}대, 1초 창 {d['windows']:,}개")
    st.markdown('<p class="head">이상 감지 방법 비교</p>', unsafe_allow_html=True)
    rows = [{"방법": v["name"], "AUC (고장 0.3↑)": v["auc"], "AUC (초기)": v["auc_early"], "정지 위험 전 경고": v["detected_before_fail"],
             "평균 여유(회차)": v["mean_lead_laps"], "주의 시점 고장 정도": v["mean_sev_at_caution"], "오경보 로봇": v["false_alarm_robots"]}
            for v in m["detector_comparison"].values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.image(str(C.REPORT_DIR / "fig_health_trend.png"), width="stretch")
    a, b, c2 = st.columns(3)
    a.image(str(C.REPORT_DIR / "fig_method_compare.png"), width="stretch")
    b.image(str(C.REPORT_DIR / "fig_importance.png"), width="stretch")
    c2.image(str(C.REPORT_DIR / "fig_confusion.png"), width="stretch")


# ---------------- 내 데이터 ----------------
def save_uploads(uploads):
    C.USER_DATA_DIR.mkdir(exist_ok=True)
    for up in uploads:
        path = C.USER_DATA_DIR / Path(up.name).name
        if not path.exists() or path.stat().st_size != up.size:
            path.write_bytes(up.getbuffer())
            clear_run()


def user_panel():
    """파일 넣기 → 형식 검사 → (선택) 학습. 읽은 데이터를 돌려주고, 문제가 있으면 안내 후 None을 돌려준다."""
    s = st.session_state
    with st.expander("내 데이터 넣기", expanded=not s.playing):
        st.caption(f"한 줄이 진동 샘플 하나인 CSV입니다. 필수 열은 `time`과 `left_vib` 또는 `right_vib`이고, 자세한 형식은 DATA_FORMAT.md에 있습니다. "
                   f"`{C.USER_DATA_DIR.name}` 폴더에 직접 넣어도 아래 목록에 나타납니다.")
        save_uploads(st.file_uploader("CSV 파일 올리기 (.csv, .csv.gz, .zip)", type=["csv", "gz", "zip"], accept_multiple_files=True))
        try:
            files = U.list_files(C.USER_DATA_DIR) if C.USER_DATA_DIR.exists() else []
        except U.UserDataError:
            files = []
        if not files:
            st.info("아직 넣은 파일이 없습니다.")
            return None
        names = [f.name for f in files]
        a, b, c = st.columns([2, 1, 1])
        picked = a.multiselect("사용할 파일", names, default=names, key="files", on_change=clear_run)
        fs = b.number_input("진동 샘플링 Hz (time 열이 없을 때)", 100.0, 200000.0, float(C.SENSORS["vib_fs"]), key="fs", on_change=clear_run)
        lap_s = c.number_input("1회차 길이 초 (lap 열이 없을 때)", 5.0, 3600.0, C.USER_LAP_S, key="lap_s", on_change=clear_run)
        if not picked:
            st.info("사용할 파일을 고르세요.")
            return None
        stamp = tuple((str(C.USER_DATA_DIR / n), (C.USER_DATA_DIR / n).stat().st_mtime, (C.USER_DATA_DIR / n).stat().st_size) for n in picked)
        try:
            data = load_user_data(stamp, fs, lap_s)
        except U.UserDataError as e:
            st.error(str(e))
            return None
        st.dataframe(data.summary(), hide_index=True, width="stretch")
        for w in data.warnings:
            st.warning(w, icon="⚠️")

        st.markdown('<p class="head">내 데이터로 학습 (선택)</p>', unsafe_allow_html=True)
        a, b = st.columns([3, 1], vertical_alignment="bottom")
        holdout = a.multiselect("학습에서 뺄 로봇 (성능 확인용)", list(data.robots), key="holdout")
        if b.button("이 데이터로 학습", width="stretch"):
            try:
                with st.spinner("학습 중… 데이터 양에 따라 수 분 걸릴 수 있습니다"):
                    pipe, _ = U.train_user(data, holdout)
                pipe.save(C.USER_BUNDLE_PATH)
                clear_run()
            except U.UserDataError as e:
                st.error(str(e))
        if C.USER_BUNDLE_PATH.exists():
            for n in load_pipeline(C.USER_BUNDLE_PATH).bundle.get("notes", []):
                st.caption("· " + n)
    return data


def user_controls():
    """내 데이터 조작부. 재생을 준비했으면 True"""
    s = st.session_state
    data = user_panel()
    if data is None or not data.robots:
        return False
    models = {}
    if C.USER_BUNDLE_PATH.exists():
        models["내 데이터로 학습한 모델"] = C.USER_BUNDLE_PATH
    if BUNDLE_PATH.exists():
        models["시뮬레이션으로 학습한 모델"] = BUNDLE_PATH
    if not models:
        st.info("사용할 모델이 없습니다. 위에서 [이 데이터로 학습]을 누르세요.")
        return False
    c = st.columns([1.3, 1.7, 1, 0.9, 0.9], vertical_alignment="bottom")
    robot = c[0].selectbox("로봇", list(data.robots), key="robot", on_change=clear_run)
    model = c[1].selectbox("모델", list(models), key="model", on_change=clear_run)
    c[2].selectbox("재생 속도", list(SPEEDS), key="speed")
    recs = data.robots[robot]
    if recs[-1]["lap"] < C.BASELINE_LAPS:
        st.error(f"{robot}은 {recs[-1]['lap'] + 1}회차뿐이라 진단할 수 없습니다. {C.BASELINE_LAPS + 1}회차 이상 필요합니다.")
        return False
    pipe = load_pipeline(models[model])
    if "run" not in s:
        s.run = Run(pipe, recs, recs[-1]["lap"] + 1)
    if c[3].button("⏸ 일시정지" if s.playing else "▶ 시작", type="primary", width="stretch", disabled=s.run.done):
        s.playing = not s.playing
        st.rerun()
    c[4].button("↺ 처음부터", width="stretch", on_click=clear_run)
    if pipe.bundle["source"] != "user":
        st.warning("시뮬레이션으로 학습한 모델입니다. 건강 점수 눈금과 원인 판단은 실제 데이터에 맞춘 것이 아니니 참고용으로 보세요.", icon="⚠️")
    if robot in pipe.bundle.get("trained_robots", []):
        st.caption(f"{robot}은 이 모델의 학습에 쓰인 로봇이라 결과를 실제 성능으로 보면 안 됩니다.")
    return True


def sim_controls():
    s = st.session_state
    if not BUNDLE_PATH.exists():
        st.error("모델 파일이 없습니다. amr_pdm 폴더에서 `python run_all.py`를 먼저 실행하세요.")
        return False
    c = st.columns([1.3, 1, 1.1, 1, 0.9, 0.9], vertical_alignment="bottom")
    c[0].selectbox("고장 시나리오", list(KINDS), key="kind", on_change=clear_run)
    c[1].selectbox("고장 위치", list(WHEELS), key="wheel", on_change=clear_run)
    c[2].selectbox("고장 진행 속도", list(RAMPS), key="ramp", on_change=clear_run)
    c[3].selectbox("재생 속도", list(SPEEDS), key="speed")
    if "run" not in s:
        s.run = simulated_run()
    if c[4].button("⏸ 일시정지" if s.playing else "▶ 시작", type="primary", width="stretch", disabled=s.run.done):
        s.playing = not s.playing
        st.rerun()
    c[5].button("↺ 처음부터", width="stretch", on_click=clear_run)
    return True


# ---------------- 페이지 ----------------
head, note = st.columns([3, 2], vertical_alignment="bottom")
head.title("AMR 고장예지 AI")

s = st.session_state
s.setdefault("playing", False)
mode = st.radio("입력 데이터", ["시뮬레이션", "내 데이터"], horizontal=True, key="mode", on_change=clear_run)
if mode == "내 데이터":
    note.caption("저장된 AI 파이프라인이 1초 창마다 실제로 추론합니다 · 입력은 사용자가 넣은 기록을 재생합니다")
    ready = user_controls()
else:
    note.caption("저장된 AI 파이프라인이 1초 창마다 실제로 추론합니다 · 입력은 시뮬레이션 데이터 (실측 검증 전)")
    ready = sim_controls()

tab_live, tab_report = st.tabs(["실시간 진단", "성능 보고서 (시뮬레이션)"])
with tab_live:
    if ready:
        st.fragment(live_view, run_every=TICK_S if s.playing else None)()
    else:
        st.info("데이터와 모델을 준비하면 여기에서 진단을 재생합니다.")
with tab_report:
    report_view()
