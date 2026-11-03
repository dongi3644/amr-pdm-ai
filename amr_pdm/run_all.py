"""전체 실행: 데이터 생성 → 학습 → 평가

    python run_all.py            # 데이터가 있으면 재사용
    python run_all.py --regen    # 데이터 다시 생성
"""
import sys

from pipeline.train import main

if __name__ == "__main__":
    main(force_data="--regen" in sys.argv)
