"""pipeline 패키지: AMR 고장예지 AI의 본체 (실측 데이터 전용)

[파일 지도] 데이터가 흘러가는 순서대로

    dataio.py    CSV 파일 → 1초 창(dict) 목록          "데이터를 읽어 자르는 곳"
        ↓
    features.py  1초 창 → 숫자 특징 (4륜 70개)             "신호를 숫자로 요약하는 곳"
        ↓
    models.py    AI 모델 부품 9종                      "각 모델의 정의"
        ↓
    system.py    부품을 하나로 묶어 학습·저장           "모델 묶음(AMRPipeline)"
        ↓
    infer.py     1초 창을 하나씩 넣으며 실시간 판정      "진단기(Predictor)"
        ↓
    workflow.py  학습 / 진단 / 로봇 단위 교차검증 절차   "실행 순서를 짜는 곳"
        ↓
    report.py    평가 결과 → 그래프·요약 문서           "결과를 보여 주는 곳"

[바깥에서 부르는 곳]
    run.py               명령줄 입구 (check / train / diagnose / paderborn)
    실행/*.py             VS Code·Visual Studio에서 버튼으로 run.py를 부르는 스크립트
    datasets/paderborn.py 공개 데이터(.mat)를 dataio가 읽는 CSV 형식으로 바꾸는 변환기

[학습할 때]  dataio → features → system.fit(모델 학습) → artifacts/model_bundle.joblib 저장
[진단할 때]  dataio → infer.Predictor.update(창마다) → 주기별 상태·원인·남은 시간
[평가할 때]  workflow.cross_validate: 로봇 한 대(또는 몇 대)를 빼고 학습 → 뺀 로봇을 진단 → 모든 로봇에 반복
"""
