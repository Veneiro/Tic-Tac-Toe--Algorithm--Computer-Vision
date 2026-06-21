#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// -------------------- SPI AMT203S-V --------------------
#define PIN_SCLK 12
#define PIN_MOSI 11
#define PIN_MISO 13
/*
#define PIN_CS1  10
#define PIN_CS2  14
#define PIN_CS3  15
*/

/*
#define PIN_CS1  15
#define PIN_CS2  10
#define PIN_CS3  14
*/
#define PIN_CS1  10
#define PIN_CS2  14
#define PIN_CS3  15

// ===================== I2C =====================
#define SDA_PIN 8
#define SCL_PIN 9
#define ADDRESS 0x34
// ===================== SERVO =====================
#define PIN_GRIPPER 18
// ===================== CONFIG =====================
#define MOTOR_FIXED_SPEED_ADDR 51
#define MOTOR_FIXED_PWM_ADDR 31
#define MOTOR_ENCODER_TOTAL_ADDR 60
#define MOTOR_TYPE_ADDR 20
#define MOTOR_ENCODER_POLARITY_ADDR 21
#define Ts 0.05 // 50 ms
#define COUNTS_PER_REV_OUTPUT 3960.0
#define TsDriver 0.01 // 10 ms
// Longitudes del robot
#define L1 25.0
#define L2 26.0
#define L3 9.65

#define HERMITE_ARC_SAMPLES 50
#define MAX_TRAJECTORY_SEGMENTS 5
#define V_max 7.5 // cm/s
#define T_acc 1.0   // s
#define z_trabajo -7.5
#define z_elevacion 5.0
// ===================== DEFINICIONES =====================
//BASE
//#define E1_MIN 32.87     //-90
//#define E1_ZERO 122.87   //(RELACION DIRECTA CON EL ENCODER RELATIVO, RELACION DIRECTA CON EL MODELO)
//#define E1_MAX 212.87    //90
#define Q1_MIN -90
#define Q1_MAX 90
//HOMBRO
//#define E2_MIN 159.43   //-8
//#define E2_ZERO 167.52  //(RELACION DIRECTA CON EL ENCODER RELATIVO, RELACION DIRECTA CON EL MODELO)
//#define E2_MAX 226.49   //51
#define Q2_MIN -8
#define Q2_MAX 51
//CODO
//#define E3_MIN 22.50   //-37
//#define E3_ZERO 77.08  //(RELACION INVERSA CON EL ENCODER RELATIVO, RELACION INVERSA CON EL MODELO)
//#define E3_MAX 114.52   //45
#define Q3_MIN -37
#define Q3_MAX 45
// ===================== ESTRUCTURAS =====================

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

struct BoxWorkspace
{
    float xMin;
    float xMax;

    float yMin;
    float yMax;

    float zMin;
    float zMax;
};

// ===================== VARIABLES EXTERNAS =====================
extern int32_t motor_encoder[4];
extern int32_t motor_encoder_prev[4];
extern float joint_encoder[3];
extern float joint_encoder_raw[3];
extern float joint_encoder_raw_prev[3];
extern float ramal_encoder[3];
extern float velocidad_motor[4];
//extern float posicion_rel[4];
//extern float posicion_offset[4];
extern float posicion_error[4];
extern float int_posicion_error[4];
extern float Kp[4];
extern float Ki[4];
extern float relacion_transmicion[3];
extern Robot my_robot;
extern LinearPosition target_position;
extern AngularPosition target_angle;
extern const LinearPosition home_position;
extern const LinearPosition array_piezas[5];
extern const LinearPosition board_grids[3][3];
extern const int pines_CS[3];
extern volatile bool interrupt_flag;
extern bool ok;
extern bool encoder_inicia_bien;
extern float joint_referencia_calibracion[3];
extern float encoder_referencia_calibracion[3];

// ===================== FUNCIONES =====================

void inicilizacion(void);
void processSerialCommand(String input);
void printHelp(void);
void applyInverseKinematicsToTarget(void);
bool isInside(const BoxWorkspace &workspace, const LinearPosition &p);
void goHome();

// ===================== CLASES =====================

class HermiteSegment {
private:
  LinearPosition P0;
  LinearPosition P1;
  LinearPosition T0;
  LinearPosition T1;

  float u_table[HERMITE_ARC_SAMPLES];
  float s_table[HERMITE_ARC_SAMPLES];

  float length;
  bool tableReady;

  float distance3D(LinearPosition a, LinearPosition b);

public:
  HermiteSegment();

  HermiteSegment(LinearPosition p0,
                 LinearPosition p1,
                 LinearPosition t0,
                 LinearPosition t1);

  void setPoints(LinearPosition p0,
                 LinearPosition p1,
                 LinearPosition t0,
                 LinearPosition t1);

  LinearPosition evaluate(float u);

  LinearPosition derivative(float u);

  void buildArcLengthTable();

  float getLength();

  float getUFromNormalizedArc(float s_norm);

  LinearPosition evaluateByNormalizedArc(float s_norm);

  LinearPosition evaluateByDistance(float s_local);
};

class TrajectoryPath {
private:
  HermiteSegment segments[MAX_TRAJECTORY_SEGMENTS];

  float segmentStart[MAX_TRAJECTORY_SEGMENTS];
  float segmentEnd[MAX_TRAJECTORY_SEGMENTS];

  int numSegments;
  float totalLength;
  bool pathReady;

public:
  TrajectoryPath();

  void clear();

  bool addSegment(HermiteSegment segment);

  void build();

  float getTotalLength();

  int getNumSegments();

  LinearPosition evaluateByDistance(float s_global);

  LinearPosition evaluateByNormalizedDistance(float s_norm);
};

class SinusoidalSCurveProfile {
private:
  float S;      // longitud total
  float Vmax;   // velocidad maxima
  float Ta;     // tiempo de aceleracion
  float Tc;     // tiempo de velocidad constante
  float Td;     // tiempo de desaceleracion
  float Ttotal; // tiempo total

public:
  SinusoidalSCurveProfile();

  SinusoidalSCurveProfile(float totalDistance,
                          float maxVelocity,
                          float accelTime);

  void setProfile(float totalDistance,
                  float maxVelocity,
                  float accelTime);

  MotionState evaluate(float t);

  float getTotalTime();

  float getTotalDistance();

  float getMaxVelocity();

  float getAccelTime();

  float getConstantTime();

  bool isFinished(float t);
};

#endif