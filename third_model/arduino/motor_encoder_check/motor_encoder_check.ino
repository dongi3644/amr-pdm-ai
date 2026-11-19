// 모터·엔코더 점검 스케치: 부품을 하나씩 따로 돌려 보고, 문제가 배선(하드웨어)인지 코드인지 가린다
//
// [쓰는 법]
//   1. 이 스케치를 올리고 시리얼 모니터를 연다. 오른쪽 아래를 115200 baud, '줄 끝 없음'(No line ending)으로 맞춘다
//   2. 처음에는 바퀴를 바닥에서 띄운다 (차체 밑에 상자를 받침)
//   3. 입력창에 글자 하나를 보내면 그 시험을 한다
//        e  엔코더만: 모터를 끄고 10초 동안 엔코더 핀 상태와 펄스 수를 보여 준다. 그동안 바퀴를 손으로 돌린다
//        1  왼쪽 모터만 전진 2초      2  오른쪽 모터만 전진 2초
//        3  왼쪽 모터만 후진 2초      4  오른쪽 모터만 후진 2초
//        p  명령 크기를 0 → 255로 올리며 양쪽 바퀴가 실제로 도는지(펄스 수) 본다 = 바퀴가 움직이기 시작하는 최소 명령 찾기
//        s  모두 정지
//   4. 시험마다 결과 줄의 '펄스'가 0인지, 부호가 맞는지 보면 어디가 문제인지 보인다 (arduino/README.md의 점검표 참고)
//
// 핀은 우리가 쓰는 코드와 같다: L298N ENA 9, IN1 7, IN2 8 / ENB 10, IN3 11, IN4 12, 엔코더 왼쪽 A 3·B 4, 오른쪽 A 2·B 5

const int ENA = 9,  IN1 = 7,  IN2 = 8;    // 왼쪽 모터
const int ENB = 10, IN3 = 11, IN4 = 12;   // 오른쪽 모터
const int L_A = 3, L_B = 4;               // 왼쪽 앞 엔코더
const int R_A = 2, R_B = 5;               // 오른쪽 앞 엔코더

volatile long leftCount = 0;
volatile long rightCount = 0;

void leftISR()  { if (digitalRead(L_B)) leftCount++;  else leftCount--; }
void rightISR() { if (digitalRead(R_B)) rightCount++; else rightCount--; }

// 4바이트 값을 읽는 도중에 인터럽트가 값을 바꾸면 엉뚱한 숫자가 나올 수 있으므로 잠깐 인터럽트를 막고 읽는다
void readCounts(long &l, long &r) {
  noInterrupts();
  l = leftCount;  r = rightCount;
  interrupts();
}

void setMotor(int left, int right) {
  digitalWrite(IN1, left > 0);   digitalWrite(IN2, left < 0);
  analogWrite(ENA, abs(left));
  digitalWrite(IN3, right > 0);  digitalWrite(IN4, right < 0);
  analogWrite(ENB, abs(right));
}

// 모터를 ms 동안 돌리고, 그동안 바뀐 펄스 수를 한 줄로 알려 준다
void runFor(const char *name, int left, int right, unsigned long ms) {
  long l0, r0, l1, r1;
  readCounts(l0, r0);
  setMotor(left, right);
  delay(ms);
  setMotor(0, 0);
  delay(300);                                   // 관성으로 조금 더 도는 것까지 셈
  readCounts(l1, r1);
  Serial.print(name);
  Serial.print("  명령 L "); Serial.print(left); Serial.print(" R "); Serial.print(right);
  Serial.print("  →  펄스 L "); Serial.print(l1 - l0);
  Serial.print("  R "); Serial.println(r1 - r0);
}

void encoderOnly() {
  setMotor(0, 0);
  Serial.println("엔코더 시험: 10초 동안 바퀴를 손으로 돌리세요 (A·B는 핀 상태 0/1)");
  unsigned long start = millis();
  while (millis() - start < 10000) {
    long l, r;
    readCounts(l, r);
    Serial.print("L 펄스 "); Serial.print(l);
    Serial.print(" (A "); Serial.print(digitalRead(L_A)); Serial.print(" B "); Serial.print(digitalRead(L_B));
    Serial.print(")   R 펄스 "); Serial.print(r);
    Serial.print(" (A "); Serial.print(digitalRead(R_A)); Serial.print(" B "); Serial.print(digitalRead(R_B));
    Serial.println(")");
    delay(250);
  }
  Serial.println("엔코더 시험 끝");
}

void sweep() {
  Serial.println("명령 크기 올리기: 양쪽 전진, 단계마다 1초");
  for (int pwm = 0; pwm <= 255; pwm += 25) {
    char name[16];
    snprintf(name, sizeof(name), "PWM %3d", pwm);
    runFor(name, pwm, pwm, 1000);
  }
  runFor("PWM 255", 255, 255, 1000);
  Serial.println("명령 크기 시험 끝 (펄스가 처음 0보다 커지는 PWM이 바퀴가 움직이기 시작하는 최소 명령)");
}

void setup() {
  Serial.begin(115200);
  int outs[] = {ENA, IN1, IN2, ENB, IN3, IN4};
  for (int i = 0; i < 6; i++) pinMode(outs[i], OUTPUT);
  setMotor(0, 0);
  pinMode(L_A, INPUT_PULLUP); pinMode(L_B, INPUT_PULLUP);
  pinMode(R_A, INPUT_PULLUP); pinMode(R_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(L_A), leftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(R_A), rightISR, RISING);
  // 이 줄이 모터를 돌릴 때마다 다시 보이면, 모터 전류 때문에 아두이노 전원이 떨어져 다시 시작되는 것이다 (전원 문제)
  Serial.println("=== 점검 스케치 시작 === (e: 엔코더, 1~4: 모터 하나씩, p: 명령 크기, s: 정지)");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  switch (c) {
    case 'e': encoderOnly(); break;
    case '1': runFor("왼쪽 전진", 200, 0, 2000); break;
    case '2': runFor("오른쪽 전진", 0, 200, 2000); break;
    case '3': runFor("왼쪽 후진", -200, 0, 2000); break;
    case '4': runFor("오른쪽 후진", 0, -200, 2000); break;
    case 'p': sweep(); break;
    case 's': setMotor(0, 0); Serial.println("정지"); break;
    default: break;                             // 줄바꿈 글자 등은 무시
  }
}
