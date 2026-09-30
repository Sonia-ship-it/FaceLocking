/*
 * NodeMCU / Wemos D1 Mini (ESP8266) Wireless Tachometer Stepper Tracker via MQTT
 * 
 * Hardware:
 *   - Wemos D1 mini / NodeMCU ESP8266
 *   - Stepper Motor: 28BYJ-48 (5V) with ULN2003 Driver
 *   - 5V Power Supply: Connected to 5V pin on Wemos D1 mini / NodeMCU Vin
 * 
 * Precision Mapping (1000 pixels <-> 1000 steps <-> 180° sweep):
 *   - Extreme Left  (9:00 o'clock):  -500 steps / 0°   (Face at left edge: ~500px)
 *   - Center        (12:00 o'clock):    0 steps / 90°  (Face at center: ~1000px)
 *   - Extreme Right (3:00 o'clock):  +500 steps / 180° (Face at right edge: ~1500px)
 *   - Resolution: 1 pixel ≈ 1 step ≈ 0.18° angular needle displacement
 * 
 * Wiring (Wemos D1 Mini / NodeMCU -> ULN2003 Driver):
 *   - Pin D1 (GPIO 5)  -> IN1
 *   - Pin D2 (GPIO 4)  -> IN2
 *   - Pin D5 (GPIO 14) -> IN3
 *   - Pin D6 (GPIO 12) -> IN4
 *   - Driver + / VCC   -> 5V (5V pin of Wemos / Vin of NodeMCU)
 *   - Driver - / GND   -> GND (Common ground)
 * 
 * Required Arduino Libraries (Install via Arduino Library Manager):
 *   1. "PubSubClient" by Nick O'Leary
 *   2. "AccelStepper" by Mike McCauley
 *   3. "ESP8266WiFi" (built-in with ESP8266 board package)
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// ================= USER CONFIGURATION =================
const char* WIFI_SSID     = "Main Hall";
const char* WIFI_PASSWORD = "Meeting@2024";

// IP Address of PC running Mosquitto Broker (e.g. 10.12.72.167)
const char* MQTT_BROKER   = "10.12.72.167";
const int   MQTT_PORT     = 1883;

// MQTT Topics
const char* TOPIC_SUB_ANGLE  = "tachometer/angle";   // Incoming number from Python model
const char* TOPIC_PUB_STATUS = "tachometer/status";  // Telemetry back to broker

// ================= STEPPER CONFIGURATION =================
// 28BYJ-48 has 2048 steps per 360 deg in half-step mode (or ~4096 in some gearboxes).
// For 180 degrees sweep: 2048 / 2 = 1024 steps.
// Adjust TOTAL_STEPS_180_DEG if your needle sweeps further or less.
const long TOTAL_STEPS_180_DEG = 1024; 

// Pin definitions (Direct GPIO numbers work universally across all board selections):
//   NodeMCU / Wemos D1 Mini Pin D1 -> GPIO 5
//   NodeMCU / Wemos D1 Mini Pin D2 -> GPIO 4
//   NodeMCU / Wemos D1 Mini Pin D5 -> GPIO 14
//   NodeMCU / Wemos D1 Mini Pin D6 -> GPIO 12
#define IN1 5   // Pin D1 (GPIO5)
#define IN2 4   // Pin D2 (GPIO4)
#define IN3 14  // Pin D5 (GPIO14)
#define IN4 12  // Pin D6 (GPIO12)

// AccelStepper 4-wire half-step mode (sequence: IN1, IN3, IN2, IN4)
AccelStepper stepper(AccelStepper::HALF4WIRE, IN1, IN3, IN2, IN4);

WiFiClient espClient;
PubSubClient mqttClient(espClient);

long targetStep = TOTAL_STEPS_180_DEG / 2; // Default to center (90 deg)
int lastReceivedAngle = 90;

void setupWifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected successfully!");
  Serial.print("NodeMCU IP Address: ");
  Serial.println(WiFi.localIP());
}

// Convert angle (0° to 180°) into stepper step position
long angleToSteps(float angleDeg) {
  // Constrain angle to [0, 180]
  if (angleDeg < 0.0) angleDeg = 0.0;
  if (angleDeg > 180.0) angleDeg = 180.0;

  // Linear mapping: 0 deg -> 0 steps, 180 deg -> TOTAL_STEPS_180_DEG
  return (long)round((angleDeg / 180.0) * TOTAL_STEPS_180_DEG);
}

// MQTT Message Callback (Receives angle from Python client)
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char message[32];
  if (length >= sizeof(message)) length = sizeof(message) - 1;
  memcpy(message, payload, length);
  message[length] = '\0';

  // Parse numeric angle directly (e.g. "90", "124.5", "45")
  float angle = atof(message);
  lastReceivedAngle = (int)angle;

  // Calculate target steps
  targetStep = angleToSteps(angle);
  stepper.moveTo(targetStep);

  Serial.print(">> [MQTT IN] Angle: ");
  Serial.print(angle);
  Serial.print("° -> Step: ");
  Serial.println(targetStep);
}

void reconnectMqtt() {
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection to ");
    Serial.print(MQTT_BROKER);
    Serial.print("...");

    String clientId = "NodeMCU-Tachometer-" + String(ESP.getChipId(), HEX);
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println(" CONNECTED!");
      mqttClient.subscribe(TOPIC_SUB_ANGLE);
      mqttClient.publish(TOPIC_PUB_STATUS, "Tachometer NodeMCU Online & Calibrated");
      Serial.print("Subscribed to topic: ");
      Serial.println(TOPIC_SUB_ANGLE);
    } else {
      Serial.print(" FAILED, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" - Retrying in 3 seconds...");
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n==========================================");
  Serial.println("  NodeMCU MQTT Tachometer Stepper Tracker ");
  Serial.println("==========================================");

  // Stepper motion profiles
  stepper.setMaxSpeed(1200.0);       // Max steps per second
  stepper.setAcceleration(2500.0);   // Acceleration in steps/s^2

  // Startup Calibration / Tachometer Needle Sweep:
  // Sweep needle 0° -> 180° -> 90° (Center)
  Serial.println("Calibrating needle sweep...");
  stepper.setCurrentPosition(0); // Assume power-on needle is resting at 0 deg (or manual home)
  stepper.runToNewPosition(TOTAL_STEPS_180_DEG);      // Sweep to 180 deg
  delay(200);
  stepper.runToNewPosition(TOTAL_STEPS_180_DEG / 2);  // Return to 90 deg center
  delay(200);
  Serial.println("Tachometer centered at 90°");

  // WiFi & MQTT Setup
  setupWifi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
}

unsigned long lastStatusTime = 0;

void loop() {
  if (!mqttClient.connected()) {
    reconnectMqtt();
  }
  mqttClient.loop();

  // Run stepper motor continuously towards targetStep with acceleration
  stepper.run();

  // Periodic status telemetry back to MQTT broker every 5 seconds
  if (millis() - lastStatusTime > 5000) {
    lastStatusTime = millis();
    char buf[64];
    snprintf(buf, sizeof(buf), "Angle: %d deg, Step: %ld/%ld", 
             lastReceivedAngle, stepper.currentPosition(), TOTAL_STEPS_180_DEG);
    mqttClient.publish(TOPIC_PUB_STATUS, buf);
  }
}
