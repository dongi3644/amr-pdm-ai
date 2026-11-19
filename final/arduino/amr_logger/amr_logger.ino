// AMR 고장예지 데이터 기록기
// MPU6050(진동 6축) + INA219(전류·배터리 전압)를 읽어 USB 시리얼로 한 줄씩 보낸다.
// PC에서는 amr_pdm_real/실행/0_아두이노_기록.py가 이 줄들을 받아 단위를 바꾸고 data 폴더에 CSV로 저장한다.
//
// [연결] 아두이노 우노 기준. 두 모듈을 같은 I2C 선에 나란히 연결한다
//   MPU6050 : VCC → 5V, GND → GND, SDA → A4, SCL → A5   (AD0 핀은 연결하지 않거나 GND = 주소 0x68)
//   INA219  : VCC → 5V, GND → GND, SDA → A4, SCL → A5   (주소 0x40, 모듈의 A0·A1 점퍼는 그대로)
//             VIN+ ← 배터리 +,  VIN- → 모터 드라이버(L298N/TB6612)의 전원 +   (배터리와 드라이버 사이에 끼움)
//             배터리 -와 아두이노 GND는 서로 연결되어 있어야 전압이 바르게 읽힌다
//   MPU6050은 차체에 단단히 고정한다 (흔들리게 붙이면 센서 자체의 흔들림이 진동으로 기록됨). 붙인 방향을 적어 둘 것
//
// [보내는 한 줄]  t_us,ax,ay,az,gx,gy,gz,shunt,bus
//   센서의 원래 정수값을 그대로 보낸다 (아두이노에서 소수 계산을 하지 않아 빠르다). 단위 변환은 PC 쪽(pipeline/arduino.py)이 한다.
//   shunt, bus(INA219)는 INA_EVERY줄에 한 번만 읽고, 나머지 줄은 빈칸으로 보낸다.
//
// [설정을 바꿀 때] 아래 센서 범위를 바꾸면 pipeline/arduino.py의 단위 변환 상수도 같이 바꿔야 한다.

#include <Wire.h>

const uint8_t MPU = 0x68;            // MPU6050 I2C 주소
const uint8_t INA = 0x40;            // INA219 I2C 주소
const unsigned long BAUD = 500000;   // 시리얼 속도. PC 쪽 BAUD와 같아야 함
const uint32_t PERIOD_US = 2000;     // 샘플 간격 2 ms = 500 Hz (진동 샘플링)
const uint8_t INA_EVERY = 10;        // INA219는 10줄에 한 번 = 50 Hz (전류·전압은 천천히 변하므로 충분)

void writeReg8(uint8_t dev, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void writeReg16(uint8_t dev, uint8_t reg, uint16_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.write(val >> 8);              // 높은 바이트 먼저
  Wire.write(val & 0xFF);
  Wire.endTransmission();
}

uint16_t readReg16(uint8_t dev, uint8_t reg) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(dev, (uint8_t)2);
  uint16_t hi = Wire.read(), lo = Wire.read();
  return (hi << 8) | lo;
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin();
  Wire.setClock(400000);             // I2C 400 kHz (빠른 모드): 한 번 읽는 데 약 0.4 ms

  // MPU6050 설정
  writeReg8(MPU, 0x6B, 0x00);        // PWR_MGMT_1: 잠자기 해제
  writeReg8(MPU, 0x1A, 0x01);        // CONFIG: 저역통과 필터 약 184 Hz (500 Hz로 읽을 때 250 Hz 넘는 떨림이 섞여 들지 않게)
  writeReg8(MPU, 0x1B, 0x08);        // GYRO_CONFIG: ±500 °/s  → 65.5 = 1 °/s
  writeReg8(MPU, 0x1C, 0x08);        // ACCEL_CONFIG: ±4 g     → 8192 = 1 g (±2 g는 문턱 충격에서 잘릴 수 있음)

  // INA219 설정: 버스 32 V 범위, 션트 ±320 mV(0.1 Ω에서 ±3.2 A), 12비트, 연속 측정 (전원을 켰을 때의 기본값과 같음)
  writeReg16(INA, 0x00, 0x399F);

  delay(100);
  Serial.println("t_us,ax,ay,az,gx,gy,gz,shunt,bus");   // 머리줄 (PC 쪽은 이 줄을 건너뜀)
}

void loop() {
  static uint32_t next = micros();
  static uint8_t count = 0;

  while ((int32_t)(micros() - next) < 0) {}   // 다음 샘플 시각까지 기다림 (일정한 간격 = 정확한 주파수 분석)
  next += PERIOD_US;
  if ((int32_t)(micros() - next) > (int32_t)PERIOD_US) next = micros();   // 많이 밀렸으면 따라잡지 않고 다시 맞춤

  uint32_t t = micros();

  // MPU6050: 0x3B부터 14바이트 = 가속도 X·Y·Z, 온도, 각속도 X·Y·Z (각 2바이트, 부호 있음)
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)14);
  int16_t v[7];
  for (uint8_t i = 0; i < 7; i++) {
    int16_t hi = Wire.read(), lo = Wire.read();
    v[i] = (hi << 8) | lo;
  }

  Serial.print(t);
  Serial.print(',');  Serial.print(v[0]);      // ax
  Serial.print(',');  Serial.print(v[1]);      // ay
  Serial.print(',');  Serial.print(v[2]);      // az
  Serial.print(',');  Serial.print(v[4]);      // gx  (v[3]은 온도라 보내지 않음)
  Serial.print(',');  Serial.print(v[5]);      // gy
  Serial.print(',');  Serial.print(v[6]);      // gz
  Serial.print(',');
  if (count == 0) {                            // INA219: 션트 전압(0x01, 부호 있음), 버스 전압(0x02)
    int16_t shunt = (int16_t)readReg16(INA, 0x01);
    uint16_t bus = readReg16(INA, 0x02);
    Serial.print(shunt);
    Serial.print(',');
    Serial.print(bus);
  } else {
    Serial.print(',');                         // 읽지 않은 줄은 빈칸 두 개
  }
  Serial.println();
  count = (count + 1) % INA_EVERY;
}
