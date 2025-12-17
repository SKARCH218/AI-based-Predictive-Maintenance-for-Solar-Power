// 기존 라이브러리 유지
#include <Wire.h>
#include <INA219_WE.h>
#include <ArduinoJson.h>

// 최대 4개의 INA219 주소(Adafruit 표준 점프 설정)
const uint8_t SENSOR_COUNT = 4;
const uint8_t SENSOR_ADDR[SENSOR_COUNT] = {0x40, 0x41, 0x44, 0x45};

// 각 센서 인스턴스 (명시적으로 생성)
INA219_WE sensor0(SENSOR_ADDR[0]);
INA219_WE sensor1(SENSOR_ADDR[1]);
INA219_WE sensor2(SENSOR_ADDR[2]);
INA219_WE sensor3(SENSOR_ADDR[3]);
INA219_WE* sensors[SENSOR_COUNT] = { &sensor0, &sensor1, &sensor2, &sensor3 };

// 센서 활성 여부 & 누적 발전량(mWh)
bool sensorActive[SENSOR_COUNT] = {false, false, false, false};
float accumulatedEnergy_mWh[SENSOR_COUNT] = {0, 0, 0, 0};

// 센서별 축 라벨(x1, x2, y1, y2 등) — 설치 구성에 맞게 수정
const char* AXIS_LABELS[SENSOR_COUNT] = { "x1", "x2", "y1", "y2" };

// 측정 간격 (1초)
const unsigned long interval = 1000;
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // 각 센서 초기화 시도
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    if (sensors[i]->init()) {
      sensorActive[i] = true;
      Serial.print("INA219 @0x");
      Serial.print(SENSOR_ADDR[i], HEX);
      Serial.println(" 연결 성공");
    } else {
      sensorActive[i] = false;
      Serial.print("INA219 @0x");
      Serial.print(SENSOR_ADDR[i], HEX);
      Serial.println(" 연결 실패 (배선/주소 확인)");
    }
  }
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
      if (!sensorActive[i]) continue;

      // 센서 측정
      float busVoltage   = sensors[i]->getBusVoltage_V();     // V
      float shuntVoltage = sensors[i]->getShuntVoltage_mV();  // mV
      float current_mA   = sensors[i]->getCurrent_mA();       // mA
      float loadVoltage  = busVoltage + (shuntVoltage / 1000.0); // V
      float power_mW     = loadVoltage * current_mA;          // mW

      // 센서별 누적 발전량 계산 (1초 단위 → mWh)
      accumulatedEnergy_mWh[i] += power_mW / 3600.0;

      // JSON 생성(한 줄당 한 센서)
      String jsonOutput;
      jsonOutput.reserve(200);
      jsonOutput = "{";
      jsonOutput += "\"sensorAddress\": \"0x" + String(SENSOR_ADDR[i], HEX) + "\",";
      jsonOutput += "\"axis\": \"" + String(AXIS_LABELS[i]) + "\",";
      jsonOutput += "\"busVoltage\": " + String(busVoltage, 3) + ",";
      jsonOutput += "\"shuntVoltage\": " + String(shuntVoltage, 3) + ",";
      jsonOutput += "\"loadVoltage\": " + String(loadVoltage, 3) + ",";
      jsonOutput += "\"current_mA\": " + String(current_mA, 3) + ",";
      jsonOutput += "\"power_mW\": " + String(power_mW, 3) + ",";
      jsonOutput += "\"accumulatedEnergy_mWh\": " + String(accumulatedEnergy_mWh[i], 6);
      jsonOutput += "}";

      // 시리얼로 전송 (파이썬 수집기는 한 줄 JSON 기준으로 처리)
      Serial.println(jsonOutput);
    }
  }
}
