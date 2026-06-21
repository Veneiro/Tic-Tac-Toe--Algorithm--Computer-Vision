#include "config.h"
#include "motorDriver.h"
#include "encoder.h"
#include "kinematics.h"
#include <Arduino.h>
#include "gripper.h"

// ===================== CONSTANTS =====================
const LinearPosition home_position = {0.0, 26.0, 15.35};
const int pines_CS[3] = {PIN_CS1, PIN_CS2, PIN_CS3};
const LinearPosition array_piezas[5] = {
  {-7.23, 26.78, z_trabajo},
  {-3.51, 27.10, z_trabajo},
  {0.09, 27.33, z_trabajo},
  {3.51, 27.11, z_trabajo},
  {7.09, 27.42, z_trabajo}
};
const LinearPosition board_grids[3][3] = {
  {{3.50, 37.8, z_trabajo}, {3.47, 34.11, z_trabajo}, {4.00, 30.37, z_trabajo}},
  {{0.0, 37.6, z_trabajo}, {0.05, 33.99, z_trabajo}, {0.05, 29.94, z_trabajo}},
  {{-3.87, 37.4, z_trabajo}, {-4.43, 33.82, z_trabajo}, {-4.24, 29.87, z_trabajo}}
};
// ===================== VARIABLES GLOBALES =====================
bool interrupt_flag = false;
bool ok = false;

//float joint_referencia_calibracion[3] = {0.0, 0.0, 0.0};//0.0, -6.42, 57.48
float joint_referencia_calibracion[3] = {0.0, 0.0, 0.0};
float encoder_referencia_calibracion[3] = {-1.05, 98.70, -170.24};

int32_t motor_encoder[4] = {0};
int32_t motor_encoder_prev[4] = {0};

float joint_encoder[3] = {0.0};

float ramal_encoder[3] = {0.0};

float joint_encoder_raw[3] = {0.0};

float joint_encoder_raw_prev[3] = {0.0};

float velocidad_motor[4] = {0.0};

float posicion_error[4] = {0};
float int_posicion_error[4] = {0};

float Kp[4] = {0.8, 0.8, 0.8, 0.0};
float Ki[4] = {0.1, 0.1, 0.1, 0.0};

float relacion_transmicion[3] = {
  185.0 / 52.5,
  32.0 / 12.0,
  32.0 / 10.0
};

LinearPosition target_position = {0.0, 0.0, 0.0};
AngularPosition target_angle = {0.0, 0.0, 0.0};

Robot my_robot = {
  {0.0, 0.0, 0.0},   // q: q1, q2, q3
  {0.0, 0.0, 0.0},   // w: w1, w2, w3
  {0.0, 0.0, 0.0},   // p: x, y, z
  {0.0, 0.0, 0.0}    // v: v_x, v_y, v_z
};

// ===================== FUNCIONES =====================

void inicilizacion(void){
  for(int i=0;i<3;i++){
        joint_encoder_raw_prev[i] = readAngleDeg(pines_CS[i]);
        joint_encoder_raw[i] = joint_encoder_raw_prev[i];
        if (joint_encoder_raw[i] > 180.0f)
        {
            ramal_encoder[i] -= 360.0f;
        }
        joint_encoder[i] = joint_encoder_raw[i] + ramal_encoder[i];
    }
  forwardKinematics();

  target_position = home_position;
  IKResult my_solution = inverseKinematics(target_position);
  ok = my_solution.hasSolution;
  target_angle = my_solution.q;
}

//***************************************************************
// ===================== CLASE HERMITE =====================
//***************************************************************
HermiteSegment::HermiteSegment() {
  P0 = {0.0f, 0.0f, 0.0f};
  P1 = {0.0f, 0.0f, 0.0f};
  T0 = {0.0f, 0.0f, 0.0f};
  T1 = {0.0f, 0.0f, 0.0f};

  length = 0.0f;
  tableReady = false;

  for (int i = 0; i < HERMITE_ARC_SAMPLES; i++) {
    u_table[i] = 0.0f;
    s_table[i] = 0.0f;
  }
}

HermiteSegment::HermiteSegment(LinearPosition p0,
                               LinearPosition p1,
                               LinearPosition t0,
                               LinearPosition t1) {
  P0 = p0;
  P1 = p1;
  T0 = t0;
  T1 = t1;

  length = 0.0f;
  tableReady = false;

  for (int i = 0; i < HERMITE_ARC_SAMPLES; i++) {
    u_table[i] = 0.0f;
    s_table[i] = 0.0f;
  }

  buildArcLengthTable();
}

// ===================== CONFIGURACIÓN =====================

void HermiteSegment::setPoints(LinearPosition p0,
                               LinearPosition p1,
                               LinearPosition t0,
                               LinearPosition t1) {
  P0 = p0;
  P1 = p1;
  T0 = t0;
  T1 = t1;

  buildArcLengthTable();
}

// ===================== EVALUACIÓN HERMITE =====================

LinearPosition HermiteSegment::evaluate(float u) {
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;

  float u2 = u * u;
  float u3 = u2 * u;

  float h00 =  2.0f * u3 - 3.0f * u2 + 1.0f;
  float h10 =         u3 - 2.0f * u2 + u;
  float h01 = -2.0f * u3 + 3.0f * u2;
  float h11 =         u3 -        u2;

  LinearPosition p;

  p.x = h00 * P0.x + h10 * T0.x + h01 * P1.x + h11 * T1.x;
  p.y = h00 * P0.y + h10 * T0.y + h01 * P1.y + h11 * T1.y;
  p.z = h00 * P0.z + h10 * T0.z + h01 * P1.z + h11 * T1.z;

  return p;
}

// ===================== DERIVADA RESPECTO A u =====================

LinearPosition HermiteSegment::derivative(float u) {
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;

  float u2 = u * u;

  float dh00 =  6.0f * u2 - 6.0f * u;
  float dh10 =  3.0f * u2 - 4.0f * u + 1.0f;
  float dh01 = -6.0f * u2 + 6.0f * u;
  float dh11 =  3.0f * u2 - 2.0f * u;

  LinearPosition dp;

  dp.x = dh00 * P0.x + dh10 * T0.x + dh01 * P1.x + dh11 * T1.x;
  dp.y = dh00 * P0.y + dh10 * T0.y + dh01 * P1.y + dh11 * T1.y;
  dp.z = dh00 * P0.z + dh10 * T0.z + dh01 * P1.z + dh11 * T1.z;

  return dp;
}

// ===================== DISTANCIA 3D =====================

float HermiteSegment::distance3D(LinearPosition a, LinearPosition b) {
  float dx = b.x - a.x;
  float dy = b.y - a.y;
  float dz = b.z - a.z;

  return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ===================== TABLA DE LONGITUD DE ARCO =====================

void HermiteSegment::buildArcLengthTable() {
  length = 0.0f;

  u_table[0] = 0.0f;
  s_table[0] = 0.0f;

  LinearPosition p_prev = evaluate(0.0f);

  for (int i = 1; i < HERMITE_ARC_SAMPLES; i++) {
    float u = (float)i / (float)(HERMITE_ARC_SAMPLES - 1);

    LinearPosition p_curr = evaluate(u);

    length += distance3D(p_prev, p_curr);

    u_table[i] = u;
    s_table[i] = length;

    p_prev = p_curr;
  }

  if (length > 0.0f) {
    for (int i = 0; i < HERMITE_ARC_SAMPLES; i++) {
      s_table[i] = s_table[i] / length;
    }
  }

  tableReady = true;
}

// ===================== LONGITUD TOTAL =====================

float HermiteSegment::getLength() {
  if (!tableReady) {
    buildArcLengthTable();
  }

  return length;
}

// ===================== CONVERSIÓN LONGITUD NORMALIZADA → u =====================

float HermiteSegment::getUFromNormalizedArc(float s_norm) {
  if (!tableReady) {
    buildArcLengthTable();
  }

  if (s_norm <= 0.0f) return 0.0f;
  if (s_norm >= 1.0f) return 1.0f;

  for (int i = 1; i < HERMITE_ARC_SAMPLES; i++) {
    if (s_norm <= s_table[i]) {
      float s0 = s_table[i - 1];
      float s1 = s_table[i];

      float u0 = u_table[i - 1];
      float u1 = u_table[i];

      float alpha = 0.0f;

      if ((s1 - s0) > 1e-6f) {
        alpha = (s_norm - s0) / (s1 - s0);
      }

      return u0 + alpha * (u1 - u0);
    }
  }

  return 1.0f;
}

// ===================== EVALUACIÓN POR LONGITUD NORMALIZADA =====================

LinearPosition HermiteSegment::evaluateByNormalizedArc(float s_norm) {
  float u = getUFromNormalizedArc(s_norm);
  return evaluate(u);
}

// ===================== EVALUACIÓN POR DISTANCIA LOCAL =====================

LinearPosition HermiteSegment::evaluateByDistance(float s_local) {
  if (!tableReady) {
    buildArcLengthTable();
  }

  if (length <= 1e-6f) {
    return P0;
  }

  float s_norm = s_local / length;

  if (s_norm < 0.0f) s_norm = 0.0f;
  if (s_norm > 1.0f) s_norm = 1.0f;

  return evaluateByNormalizedArc(s_norm);
}
//***************************************************************
// ===================== CLASE TRAYECTORIA =====================
//***************************************************************
TrajectoryPath::TrajectoryPath() {
  numSegments = 0;
  totalLength = 0.0f;
  pathReady = false;

  for (int i = 0; i < MAX_TRAJECTORY_SEGMENTS; i++) {
    segmentStart[i] = 0.0f;
    segmentEnd[i] = 0.0f;
  }
}

void TrajectoryPath::clear() {
  numSegments = 0;
  totalLength = 0.0f;
  pathReady = false;

  for (int i = 0; i < MAX_TRAJECTORY_SEGMENTS; i++) {
    segmentStart[i] = 0.0f;
    segmentEnd[i] = 0.0f;
  }
}

bool TrajectoryPath::addSegment(HermiteSegment segment) {
  if (numSegments >= MAX_TRAJECTORY_SEGMENTS) {
    return false;
  }

  segments[numSegments] = segment;
  numSegments++;

  pathReady = false;

  return true;
}

void TrajectoryPath::build() {
  totalLength = 0.0f;

  for (int i = 0; i < numSegments; i++) {
    float L = segments[i].getLength();

    segmentStart[i] = totalLength;
    totalLength += L;
    segmentEnd[i] = totalLength;
  }

  pathReady = true;
}

float TrajectoryPath::getTotalLength() {
  if (!pathReady) {
    build();
  }

  return totalLength;
}

int TrajectoryPath::getNumSegments() {
  return numSegments;
}

LinearPosition TrajectoryPath::evaluateByDistance(float s_global) {
  if (!pathReady) {
    build();
  }

  LinearPosition p_zero = {0.0f, 0.0f, 0.0f};

  if (numSegments <= 0) {
    return p_zero;
  }

  if (totalLength <= 1e-6f) {
    return segments[0].evaluate(0.0f);
  }

  if (s_global <= 0.0f) {
    return segments[0].evaluateByDistance(0.0f);
  }

  if (s_global >= totalLength) {
    return segments[numSegments - 1].evaluateByDistance(
      segments[numSegments - 1].getLength()
    );
  }

  for (int i = 0; i < numSegments; i++) {
    if (s_global <= segmentEnd[i]) {
      float s_local = s_global - segmentStart[i];

      return segments[i].evaluateByDistance(s_local);
    }
  }

  return segments[numSegments - 1].evaluateByDistance(
    segments[numSegments - 1].getLength()
  );
}

LinearPosition TrajectoryPath::evaluateByNormalizedDistance(float s_norm) {
  if (!pathReady) {
    build();
  }

  if (s_norm < 0.0f) s_norm = 0.0f;
  if (s_norm > 1.0f) s_norm = 1.0f;

  float s_global = s_norm * totalLength;

  return evaluateByDistance(s_global);
}
//***************************************************************
// ===================== CLASE PERFIL =====================
//***************************************************************
SinusoidalSCurveProfile::SinusoidalSCurveProfile() {
  S = 0.0f;
  Vmax = 0.0f;
  Ta = 0.0f;
  Tc = 0.0f;
  Td = 0.0f;
  Ttotal = 0.0f;
}

SinusoidalSCurveProfile::SinusoidalSCurveProfile(float totalDistance,
                                                 float maxVelocity,
                                                 float accelTime) {
  setProfile(totalDistance, maxVelocity, accelTime);
}

void SinusoidalSCurveProfile::setProfile(float totalDistance,
                                         float maxVelocity,
                                         float accelTime) {
  S = totalDistance;
  Vmax = maxVelocity;
  Ta = accelTime;
  Td = accelTime;

  if (S <= 0.0f || Vmax <= 0.0f || Ta <= 0.0f) {
    S = 0.0f;
    Vmax = 0.0f;
    Ta = 0.0f;
    Tc = 0.0f;
    Td = 0.0f;
    Ttotal = 0.0f;
    return;
  }

  /*
    Para este perfil:

    Durante aceleracion:
      distancia = 0.5 * Vmax * Ta

    Durante desaceleracion:
      distancia = 0.5 * Vmax * Td

    Si Ta = Td:
      distancia aceleracion + desaceleracion = Vmax * Ta

    Entonces:
      S = Vmax * Ta + Vmax * Tc
      Tc = S/Vmax - Ta
  */

  Tc = S / Vmax - Ta;

  if (Tc < 0.0f) {
    /*
      Trayectoria corta:
      no se alcanza Vmax.
      Se usa perfil triangular suavizado.

      Con Tc = 0:
        S = Vpeak * Ta

      Por tanto:
        Vpeak = S / Ta
    */
    Tc = 0.0f;
    Vmax = S / Ta;
  }

  Ttotal = Ta + Tc + Td;
}

MotionState SinusoidalSCurveProfile::evaluate(float t) {
  MotionState state;

  state.s = 0.0f;
  state.v = 0.0f;
  state.a = 0.0f;

  if (Ttotal <= 0.0f || S <= 0.0f) {
    return state;
  }

  if (t < 0.0f) {
    t = 0.0f;
  }

  if (t > Ttotal) {
    t = Ttotal;
  }

  float Sa = 0.5f * Vmax * Ta;
  float Sc = Vmax * Tc;

  // -------------------------------
  // Fase 1: aceleracion sinusoidal
  // -------------------------------
  if (t <= Ta) {
    float u = t / Ta;

    state.a = (PI * Vmax / (2.0f * Ta)) * sinf(PI * u);

    state.v = 0.5f * Vmax * (1.0f - cosf(PI * u));

    state.s = 0.5f * Vmax *
              (t - (Ta / PI) * sinf(PI * u));
  }

  // -------------------------------
  // Fase 2: velocidad constante
  // -------------------------------
  else if (t <= Ta + Tc) {
    float tc = t - Ta;

    state.a = 0.0f;
    state.v = Vmax;
    state.s = Sa + Vmax * tc;
  }

  // -------------------------------
  // Fase 3: desaceleracion sinusoidal
  // -------------------------------
  else {
    float td = t - Ta - Tc;
    float u = td / Td;

    state.a = -(PI * Vmax / (2.0f * Td)) * sinf(PI * u);

    state.v = 0.5f * Vmax * (1.0f + cosf(PI * u));

    state.s = Sa + Sc +
              0.5f * Vmax *
              (td + (Td / PI) * sinf(PI * u));
  }

  if (state.s < 0.0f) {
    state.s = 0.0f;
  }

  if (state.s > S) {
    state.s = S;
  }

  return state;
}

float SinusoidalSCurveProfile::getTotalTime() {
  return Ttotal;
}

float SinusoidalSCurveProfile::getTotalDistance() {
  return S;
}

float SinusoidalSCurveProfile::getMaxVelocity() {
  return Vmax;
}

float SinusoidalSCurveProfile::getAccelTime() {
  return Ta;
}

float SinusoidalSCurveProfile::getConstantTime() {
  return Tc;
}

bool SinusoidalSCurveProfile::isFinished(float t) {
  return t >= Ttotal;
}