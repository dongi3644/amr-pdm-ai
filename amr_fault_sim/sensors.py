"""센서 모델: 실제 센서의 샘플링 속도·측정 범위·잡음·분해능을 흉내 냄

센서를 바꾸고 싶으면 config.SENSORS 값만 수정 (예: ADXL345 → vib_fs=3200)
"""
import numpy as np


def accelerometer(phys_sig, phys_fs, cfg, rng):
    """가속도 센서 (예: MPU6050)
    - 센서 샘플링보다 높은 주파수는 내부 저역필터로 걸러짐
    - 측정 범위를 넘으면 잘림(포화)
    """
    fs = cfg["vib_fs"]
    step = phys_fs // fs
    # 간단한 저역통과(이동평균) 후 다운샘플
    lp = np.convolve(phys_sig, np.ones(step) / step, mode="same")
    out = lp[::step]
    out = out + rng.standard_normal(len(out)) * cfg["vib_noise_g"]
    return np.clip(out, -cfg["vib_range_g"], cfg["vib_range_g"])


def current_sensor(true_current, cfg, rng):
    """전류 센서 (예: ACS712, INA219)"""
    return true_current + rng.standard_normal(np.shape(true_current)) * cfg["cur_noise_a"]


def temperature_sensor(true_temp, cfg):
    """온도 센서 (예: DS18B20) — 분해능 단위로 반올림"""
    res = cfg["temp_res"]
    return np.round(np.asarray(true_temp) / res) * res
