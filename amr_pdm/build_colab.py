"""Colab 노트북 생성: amr_pdm 소스를 노트북 안에 넣어, 파일 1개만 올리면 바로 돌아가게 한다.

    python build_colab.py      # 코드를 고친 뒤 다시 실행 → ../AMR_고장예지AI_Colab.ipynb 갱신
"""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "AMR_고장예지AI_Colab.ipynb"
SOURCES = ["config.py", "run_all.py", "run_monitor.py", "run_custom.py", "DATA_FORMAT.md", "requirements.txt", ".streamlit/config.toml",
           "dashboard/app.py", "pipeline/*.py", "sim/*.py"]


def source_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in SOURCES:
            for p in sorted(ROOT.glob(pattern)):
                z.write(p, p.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


INTRO = """# AMR 고장예지 AI (Colab)

배송로봇 구동부의 진동·전류·온도로 **어느 바퀴의 무엇이 고장 나는지** 미리 알려 주는 파이프라인입니다.

```
특징 추출 → Ridge(전류 잔차) → 구간별 z-score
                                 ├→ 오토인코더(건강 점수) → 칼만 필터(남은 회차)
                                 └→ 랜덤포레스트 + SHAP(원인·판단 근거)
```

**사용법:** 런타임은 CPU면 충분합니다. 1번 셀을 실행한 뒤 목적에 맞게 고르세요. 코드는 이 노트북에 들어 있어 다른 파일을 올릴 필요가 없습니다.

- **시뮬레이션으로 확인:** 2~4번 셀
- **내 데이터로 실행:** 5~7번 셀 (시뮬레이션 모델로 진단만 하려면 2번 셀도 필요합니다)
- **대시보드:** 8번 셀

> 2~4번 셀의 수치는 **시뮬레이션 데이터** 결과입니다. 실측 검증 전이므로 발표 수치로 그대로 쓰면 안 됩니다.
> Colab 런타임이 끊기면 파일이 사라지니, 필요한 결과는 10번 셀로 내려받으세요."""

SETUP = '''#@title 1. 코드 풀기 + 패키지 설치 (약 1~2분)
import base64, io, os, subprocess, sys, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
WORK = (Path("/content") if IN_COLAB else Path.cwd()) / "amr_pdm"
SOURCE_ZIP = "__SOURCE_ZIP__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ZIP))).extractall(WORK)

def run(args, **kw):
    """명령을 실행하며 출력을 바로 보여 준다. 실패하면 멈춘다."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", **kw)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"실패 (종료 코드 {p.returncode}): {' '.join(map(str, args))}")

if IN_COLAB:
    run(["apt-get", "-qq", "-y", "install", "fonts-noto-cjk"])   # 그래프 한글 글꼴
    run([sys.executable, "-m", "pip", "install", "-q", "shap", "streamlit==1.50.0"])
sys.path.insert(0, str(WORK))
print("준비 완료:", WORK)'''

TRAIN = '''#@title 2. 데이터 생성 + 학습 + 평가 (Colab에서는 수 분 소요)
run([sys.executable, "-u", "run_all.py"])'''

RESULT = '''#@title 3. 결과 보기
import json
import pandas as pd
from IPython.display import Image, Markdown, display

reports = WORK / "reports"
m = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
display(Markdown("### 이상 감지 방법 비교 (시뮬레이션 데이터)"))
display(pd.DataFrame(m["detector_comparison"]).T.set_index("name"))
f = m["forecast_compare"]
display(Markdown(f"- **원인 분류:** 창 단위 {m['cause_window_accuracy']:.0%}, 로봇 단위 {m['cause_robot_accuracy']}\\n"
                 f"- **정지 위험 예측 오차:** 칼만 필터 {f['kalman']['mae']}회차, 추세선 {f['trend']['mae']}회차, "
                 f"90% 범위 적중 {f['kalman']['coverage']:.0%}"))
for name in ("fig_health_trend", "fig_method_compare", "fig_importance", "fig_confusion"):
    display(Image(str(reports / f"{name}.png")))'''

LIVE = '''#@title 4. 로봇 1대 실시간 진단 (노트북 안에서 재생)
kind = "bearing"  #@param ["bearing", "looseness", "unbalance", "normal"]
wheel = "right"  #@param ["right", "left"]
repair_lap = 30  #@param {type:"integer"}
seed = 600  #@param {type:"integer"}
# repair_lap: 이 회차에 정비 완료 (0이면 정비 안 함). 베어링은 정비 후 복귀까지 약 20회차 걸립니다.

import glob
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from matplotlib import font_manager

import config as C
from pipeline.system import AMRPipeline
from sim.simulator import AMRSimulator

for p in glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):
    font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = [f for f in ["Noto Sans CJK JP", "Malgun Gothic", "AppleGothic"]
                               if f in {x.name for x in font_manager.fontManager.ttflist}] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

N_LAPS = 60
faults = [] if kind == "normal" else [dict(kind=kind, wheel=wheel, start_lap=8, end_lap=32)]
sim = AMRSimulator(faults=faults, n_laps=N_LAPS, seed=seed)
monitor = AMRPipeline.load(WORK / "artifacts" / "model_bundle.joblib").monitor()

def show(out, repaired):
    laps = [l for l, _ in monitor.lap_scores]
    scores = [s for _, s in monitor.lap_scores]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, v, c in (("주의", 3, "#fab219"), ("경고", 6, "#ec835a"), ("정지 위험", 9, "#d03b3b")):
        ax.axhline(v, color=c, lw=1, ls="--")
        ax.text(0.5, v + 0.12, f"{name} {v}", color="#52514e", fontsize=9)
    ax.plot(laps, scores, color="#2a78d6", lw=2, marker="o", ms=3)
    fc = None if repaired else monitor.forecast
    if fc and fc["laps"] > 0:
        ax.plot([laps[-1], laps[-1] + fc["laps"]], [fc["level"], 9], color="#52514e", lw=2, ls=":")
    ax.set(xlim=(0, N_LAPS), ylim=(0, 10.5), xlabel="회차", ylabel="건강 점수")
    ax.spines[["top", "right"]].set_visible(False)
    clear_output(wait=True)
    display(fig)
    plt.close(fig)
    state = C.STATUS_KO[monitor.status]
    print(f"{laps[-1]}회차 | 상태: {state} | 점수: {scores[-1]:.1f}" + (" | 정비 완료, 복귀 확인 중" if repaired else ""))
    if out and out.get("cause") and not repaired:
        print("추정 원인:", out["cause"])
        print("판단 근거:", ", ".join(n for n, _ in monitor.reason) or "-")
    if fc:
        print(f"정지 위험까지: 약 {fc['laps']:.0f}회차 (빠르면 {fc['laps_lo']:.0f}회차)")

seen, out, repaired = 0, None, False
for rec in sim.windows():
    if repair_lap and not repaired and rec["lap"] >= repair_lap:
        sim.repair()
        repaired = True
    o = monitor.update(rec)
    if o["phase"] == "monitor":
        out = o
    if len(monitor.lap_scores) > seen:
        seen = len(monitor.lap_scores)
        show(out, repaired and monitor.status > 0)
monitor.finish()
show(out, False)
print("\\n상태 변경 이력:", [(l, C.STATUS_KO[s]) for l, s in monitor.events] or "없음")'''

USER_NOTE = """## 내 데이터로 실행

**한 줄이 진동 샘플 하나**인 CSV를 파일 하나당 로봇 1대로 넣습니다. 필수 열은 `time`(초)과 `left_vib` 또는 `right_vib`(g)입니다.

```csv
time,lap,segment,speed,left_vib,right_vib,left_cur,right_cur,left_temp,right_temp
0.000,0,복도A,1.02,0.0132,-0.0071,0.84,0.81,31.2,30.9
0.001,0,복도A,1.02,0.0094,0.0023,,,,
```

- 학습에 쓰려면 라벨 열 `fault_kind`, `fault_wheel`, `fault_severity`를 넣습니다. 라벨이 없으면 모두 정상으로 보고 학습합니다.
- **로봇마다 처음 5회차는 정상**이어야 하고, 진단하려면 6회차 이상이 필요합니다.
- 전체 설명은 5번 셀을 실행한 뒤 `amr_pdm/DATA_FORMAT.md`에서 볼 수 있습니다."""

USER_LOAD = '''#@title 5. 내 데이터 넣기 + 형식 검사
drive_folder = ""  #@param {type:"string"}
make_sample = False  #@param {type:"boolean"}
lap_seconds = 60  #@param {type:"number"}
# 아무것도 안 하면 파일 선택 창이 뜹니다. 큰 파일은 드라이브 폴더 경로를 쓰는 편이 빠릅니다 (예: /content/drive/MyDrive/amr_data).
# make_sample을 켜면 시뮬레이터로 예시 파일 3개를 만듭니다 (수 분 소요). 실제 데이터를 넣을 때는 sample_ 파일을 지우세요.
# lap_seconds: 파일에 lap 열이 없을 때 1회차로 볼 시간(초)

import shutil

user_dir = WORK / "user_data"
user_dir.mkdir(exist_ok=True)
if make_sample:
    run([sys.executable, "-u", "run_custom.py", "sample"])
elif drive_folder:
    if IN_COLAB and drive_folder.startswith("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    for p in sorted(Path(drive_folder).iterdir()):
        if p.name.lower().endswith((".csv", ".csv.gz", ".zip")):
            shutil.copy(p, user_dir / p.name)
elif IN_COLAB:
    from google.colab import files
    for name, content in files.upload().items():
        (user_dir / Path(name).name).write_bytes(content)
        if Path(name).exists():
            os.remove(name)
print("넣은 파일:", [p.name for p in sorted(user_dir.iterdir())])
user_args = ["--lap-s", str(lap_seconds)]
run([sys.executable, "run_custom.py", "check"] + user_args)'''

USER_TRAIN = '''#@title 6. (선택) 내 데이터로 학습
holdout = ""  #@param {type:"string"}
# holdout: 학습에서 빼고 성능 확인에 쓸 로봇 이름 (쉼표로 구분, 예: R3,R4). 빼지 않으면 실제 성능을 알 수 없습니다.
run([sys.executable, "-u", "run_custom.py", "train", "--holdout", holdout] + user_args)'''

USER_DIAGNOSE = '''#@title 7. 내 데이터 진단 (로봇별 결과 표와 그래프)
model = "auto"  #@param ["auto", "user", "sim"]
# auto: 6번 셀로 학습한 모델이 있으면 그 모델, 없으면 시뮬레이션 모델(2번 셀 필요)
import pandas as pd
from IPython.display import Image, Markdown, display

run([sys.executable, "run_custom.py", "diagnose", "--model", model] + user_args)
for table_path in sorted((WORK / "reports" / "user").glob("*.csv")):
    display(Markdown(f"### {table_path.stem}"))
    display(Image(str(table_path.with_suffix(".png"))))
    display(pd.read_csv(table_path).fillna("").tail(10))'''

DASH = '''#@title 8. 대시보드 실행 (Colab 내부 프록시, 본인만 접속 가능)
import time, urllib.request

if "dash_proc" in globals() and dash_proc.poll() is None:
    dash_proc.terminate()
    dash_proc.wait()
dash_log = open(WORK / "streamlit.log", "w", encoding="utf-8")
dash_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501",
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
    cwd=WORK, stdout=dash_log, stderr=subprocess.STDOUT,
    env=dict(os.environ, PYTHONUTF8="1", AMR_TICK_S="1.5"))   # 프록시 지연을 고려해 갱신 주기를 늘림
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1)
        break
    except OSError:
        time.sleep(1)
else:
    print(open(WORK / "streamlit.log", encoding="utf-8").read())
    raise RuntimeError("대시보드가 시작되지 않았습니다. 위 로그를 확인하세요.")

if IN_COLAB:
    from google.colab import output
    print("아래 링크를 클릭하세요. 화면이 'Please wait...'에서 멈추면 9번 셀을 쓰세요.")
    print("내 데이터는 화면 위쪽 [입력 데이터 → 내 데이터]에서 고릅니다. 5번 셀로 넣은 파일이 목록에 나타납니다.")
    output.serve_kernel_port_as_window(8501, path="/", anchor_text="AMR 대시보드 열기")
else:
    print("대시보드: http://localhost:8501")'''

TUNNEL_NOTE = """### 9. 대시보드가 열리지 않을 때만

Colab 프록시로 Streamlit 화면이 연결되지 않으면 아래 셀을 실행하세요. Cloudflare 임시 터널로 접속 주소를 만듭니다.

> **주의:** 이 주소는 링크를 아는 사람 누구나 접속할 수 있고, 대시보드에 넣은 내 데이터도 그 사람이 볼 수 있습니다. 쓰고 나면 런타임을 끊으세요."""

TUNNEL = '''#@title 9. (선택) 임시 공개 주소로 열기
import re, time, urllib.request

if not IN_COLAB:
    print("Colab 전용 셀입니다.")
else:
    binary = WORK / "cloudflared"
    if not binary.exists():
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
        binary.chmod(0o755)
    tunnel_log = WORK / "tunnel.log"
    tunnel_proc = subprocess.Popen([str(binary), "tunnel", "--no-autoupdate", "--url", "http://localhost:8501"],
                                   stdout=open(tunnel_log, "w"), stderr=subprocess.STDOUT)
    for _ in range(40):
        found = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", tunnel_log.read_text())
        if found:
            print("대시보드 주소:", found.group(0))
            break
        time.sleep(1)
    else:
        print(tunnel_log.read_text())'''

DOWNLOAD = '''#@title 10. 결과 내려받기 (그래프·수치·모델)
import shutil

bundle = WORK.parent / "amr_results"
shutil.rmtree(bundle, ignore_errors=True)
bundle.mkdir()
for folder in ("reports", "artifacts"):
    if (WORK / folder).exists():
        shutil.copytree(WORK / folder, bundle / folder)
archive = shutil.make_archive(str(WORK.parent / "amr_results"), "zip", bundle)
if IN_COLAB:
    from google.colab import files
    files.download(archive)
else:
    print("저장:", archive)'''


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type="code", metadata=dict(cellView="form"), execution_count=None, outputs=[],
                source=text.splitlines(True))


def main():
    cells = [md(INTRO), code(SETUP.replace("__SOURCE_ZIP__", source_zip())), code(TRAIN), code(RESULT), code(LIVE),
             md(USER_NOTE), code(USER_LOAD), code(USER_TRAIN), code(USER_DIAGNOSE),
             code(DASH), md(TUNNEL_NOTE), code(TUNNEL), code(DOWNLOAD)]
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python"), colab=dict(provenance=[])))
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell{i}"
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
