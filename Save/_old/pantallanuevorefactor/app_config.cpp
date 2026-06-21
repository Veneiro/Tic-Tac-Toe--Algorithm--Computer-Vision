#include "app_contracts.h"

const char *ssid = "MIM-DPM-GRUPO-3";
const char *password = "mim-dpm-2026";

const char *raspberryPi_IP = "192.168.1.28";
const int raspberryPi_PORT = 5000;

WebServer server(80);

const int pinJoyX = 1;
const int pinJoyY = 2;
const int pinJoyButton = 3;

LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);

const int pinLedRojo = 4;
const int pinLedAmarillo = 5;
const int pinLedVerde = 6;

const int pinSwitch = 36;
const int pinStart = 35;
const int pinMenu = 42;

const int pinSDA = 8;
const int pinSCL = 9;

const int PIN_SCLK = 12;
const int PIN_MOSI = 11;
const int PIN_MISO = 13;

const int PIN_CS1 = 10;
const int PIN_CS2 = 14;
const int PIN_CS3 = 15;

const int PIN_GRIPPER = 16;

const double MOTOR_FIXED_SPEED_ADDR = 51;
const double MOTOR_FIXED_PWM_ADDR = 31;
const double MOTOR_ENCODER_TOTAL_ADDR = 60;
const double MOTOR_TYPE_ADDR = 20;
const double MOTOR_ENCODER_POLARITY_ADDR = 21;

const double Ts = 0.05;
const double COUNTS_PER_REV_OUTPUT = 3960.0;
const double TsDriver = 0.01;

const double L1 = 25.0;
const double L2 = 26.0;
const double L3 = 9.65;

const double HERMITE_ARC_SAMPLES = 50;
const double MAX_TRAJECTORY_SEGMENTS = 5;
const double V_max = 10.0;
const double T_acc = 1.0;
const double z_trabajo = -7.0;
const double z_elevacion = 3.0;

const int Q1_MIN = -90;
const int Q1_MAX = 90;

const int Q2_MIN = -8;
const int Q2_MAX = 51;

const int Q3_MIN = -37;
const int Q3_MAX = 45;

struct AngularPosition {
  float q1;
  float q2;
  float q3;
};

struct AngularVelocity {
  float w1;
  float w2;
  float w3;
};

struct LinearPosition {
  float x;
  float y;
  float z;
};

struct LinearVelocity {
  float v_x;
  float v_y;
  float v_z;
};

struct Robot {
  AngularPosition q;
  AngularVelocity w;
  LinearPosition p;
  LinearVelocity v;
};

struct IKResult {
  bool hasSolution;
  bool withinLimits;
  AngularPosition q;
};

struct MotionState {
  float s;  // distancia recorrida
  float v;  // velocidad
  float a;  // aceleracion
};