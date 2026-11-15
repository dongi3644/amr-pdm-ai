// 팀 기록 스케치 (최종, 2026-11-15) — MPU6050 6축 + INA219 + 모터 실드 v2 + 엔코더 1개, 500 Hz
// 예전 버전(가속도만 · 250000 baud)과 달라진 곳에 [수정] 표시. 출력 형식은 second_model의 Python 코드가 그대로 읽는다.
//   한 줄: t_us,ax,ay,az,gx,gy,gz,mV,mA,spd,enc1   (센서 원래 정수값. Python이 ÷8192 → g, ÷65.5 → °/s로 바꿈)
//   '#'로 시작하는 줄 = 상태 문구 (Python이 건너뜀)
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS;
Adafruit_DCMotor *mt[4];
Adafruit_INA219 ina;
bool shieldOk, inaOk;

// [수정] 엔코더가 달린 바퀴의 열 이름 (Python: enc1 앞왼 · enc2 앞오른 · enc3 뒤왼 · enc4 뒤오른)
// M1·M4가 오른쪽 바퀴이므로 M2는 왼쪽 바퀴. M2가 앞 왼쪽이면 "enc1", 뒤 왼쪽이면 "enc3"으로 바꿀 것
#define ENC_COL "enc1"

// ---- 엔코더 (M2: A=D3, B=D5) ----
volatile long encCount = 0;
volatile uint8_t encPrev = 0;
const int8_t QTAB[16] = {0,-1,1,0, 1,0,0,-1, -1,0,0,1, 0,1,-1,0};

ISR(PCINT2_vect) {
  uint8_t s = (((PIND >> 3) & 1) << 1) | ((PIND >> 5) & 1);
  encCount += QTAB[(encPrev << 2) | s];
  encPrev = s;
}

void encSetup() {
  pinMode(3, INPUT_PULLUP);
  pinMode(5, INPUT_PULLUP);
  encPrev = (((PIND >> 3) & 1) << 1) | ((PIND >> 5) & 1);
  PCMSK2 |= (1 << PCINT19) | (1 << PCINT21);
  PCICR  |= (1 << PCIE2);
}

long encTake() {                          // 지난 샘플 이후 센 펄스 수 (2 ms마다 0으로 다시 셈)
  noInterrupts();
  long v = encCount;
  encCount = 0;
  interrupts();
  return v;
}

// ---- 모터 ----
int spd = 0;
void setSpeed(int level) {
  spd = level;
  for (int i = 0; i < 4; i++) {
    bool reversed = (i == 0 || i == 3);   // M1, M4 = 오른쪽 바퀴
    mt[i]->setSpeed(level * 28);
    mt[i]->run(level ? (reversed ? BACKWARD : FORWARD) : RELEASE);
  }
}

// ---- INA219 ---- [수정] loop() 안의 static 변수를 전역으로 옮기고 읽기를 함수로 묶음 (setup에서 미리 한 번 읽으려고)
long mV = 0, mA = 0;
unsigned long lastIna = 0;

void inaRead() {
  Wire.setClock(100000);
  float v = ina.getBusVoltage_V();
  float i = ina.getCurrent_mA();
  Wire.setClock(400000);
  if (v > 1.0) { mV = (long)(v * 1000); mA = (long)i; }
  lastIna = millis();
}

// ---- MPU6050 ----
const uint8_t MPU = 0x68;
int16_t ax, ay, az;
int16_t gx, gy, gz;                       // [수정] 각속도 3축 추가

void mpuInit() {
  Wire.beginTransmission(MPU); Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission();  // 절전 해제
  Wire.beginTransmission(MPU); Wire.write(0x1A); Wire.write(0x01); Wire.endTransmission();  // [수정] 저역필터 0x00 → 0x01 (약 184 Hz): 빠른 떨림이 가짜 느린 진동으로 섞이지 않게
  Wire.beginTransmission(MPU); Wire.write(0x1B); Wire.write(0x08); Wire.endTransmission();  // [수정] 자이로 범위 ±500°/s (65.5 LSB/°/s)
  Wire.beginTransmission(MPU); Wire.write(0x1C); Wire.write(0x08); Wire.endTransmission();  // 가속도 범위 ±4g (8192 LSB/g)
}

// [수정] 가속도(6) + 온도(2) + 자이로(6) = 14바이트를 한 번에 읽고, 배열에 먼저 담아 바이트 순서를 확실히 함
//        (예전 '(Wire.read() << 8) | Wire.read()'는 두 read()의 실행 순서가 C++에서 정해져 있지 않음)
bool mpuRead() {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU, (uint8_t)14, (uint8_t)true) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  ax = (int16_t)((b[0]  << 8) | b[1]);
  ay = (int16_t)((b[2]  << 8) | b[3]);
  az = (int16_t)((b[4]  << 8) | b[5]);
  // b[6], b[7] = 온도 (쓰지 않음)
  gx = (int16_t)((b[8]  << 8) | b[9]);
  gy = (int16_t)((b[10] << 8) | b[11]);
  gz = (int16_t)((b[12] << 8) | b[13]);
  return true;
}

void setup() {
  Serial.begin(500000);                   // [수정] 250000 → 500000 (한 줄이 길어져 2 ms 안에 보내려면 필요)
  Wire.begin();
  Wire.setWireTimeout(25000, true);
  shieldOk = AFMS.begin();
  for (int i = 0; i < 4; i++) mt[i] = AFMS.getMotor(i + 1);
  setSpeed(0);
  inaOk = ina.begin();
  Wire.setClock(400000);
  mpuInit();
  encSetup();
  if (inaOk) inaRead();                   // [수정] 첫 줄부터 전압·전류가 0이 아니게 미리 한 번 읽음
  Serial.println(shieldOk ? F("# shield ok") : F("# shield FAIL"));   // [수정] 상태 문구 앞에 '#'
  Serial.println(inaOk ? F("# ina ok") : F("# ina FAIL"));
  Serial.println(F("t_us,ax,ay,az,gx,gy,gz,mV,mA,spd," ENC_COL));   // [수정] gx,gy,gz 추가 · 엔코더 열 이름을 바퀴 위치에 맞춤
}

void loop() {
  if (Serial.available()) {               // PC에서 '0'~'9'를 보내면 모터 속도 단계 변경
    char c = Serial.read();
    if (c >= '0' && c <= '9') setSpeed(c - '0');
  }

  static unsigned long next = 0;
  if ((long)(micros() - next) < 0) return;
  next = micros() + 2000;                 // 2 ms마다 = 500 Hz

  if (inaOk && millis() - lastIna >= 100) inaRead();   // 0.1초마다 전압·전류

  if (!mpuRead()) { Serial.println(F("# MPU FAIL")); return; }                                 // [수정] '#'
  if (ax == 0 && ay == 0 && az == 0) { Serial.println(F("# MPU ZERO")); mpuInit(); return; }   // [수정] '#'

  Serial.print(micros()); Serial.print(',');
  Serial.print(ax); Serial.print(',');
  Serial.print(ay); Serial.print(',');
  Serial.print(az); Serial.print(',');
  Serial.print(gx); Serial.print(',');    // [수정] 각속도 3축
  Serial.print(gy); Serial.print(',');
  Serial.print(gz); Serial.print(',');
  Serial.print(mV); Serial.print(',');
  Serial.print(mA); Serial.print(',');
  Serial.print(spd); Serial.print(',');
  Serial.println(encTake());
}
