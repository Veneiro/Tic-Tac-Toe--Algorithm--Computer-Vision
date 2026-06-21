#include "config.h"
#include "motorDriver.h"
#include "encoder.h"
#include "kinematics.h"
#include <Arduino.h>
#include "gripper.h"

const LinearPosition home_position = {0.0, 26.0, 15.35};
const int pines_CS[3] = {PIN_CS1, PIN_CS2, PIN_CS3};
const LinearPosition array_piezas[5] = {
  {-8.2, 27.00, z_trabajo},
  {-4.3, 27.40, z_trabajo},
  {-1.0, 27.47, z_trabajo},
  {2.70, 27.8, z_trabajo},
  {6.3, 27.88, z_trabajo}
};
const LinearPosition array_piezas_enemigas[5] = {
  {-8.72, 40.94, z_trabajo},
  {-4.83, 41.40, z_trabajo},
  {-1.66, 41.42, z_trabajo},
  {2.00, 41.42, z_trabajo},
  {5.64, 41.91, z_trabajo}
};
const LinearPosition board_grids[3][3] = {
  {{2.9, 38.57, z_trabajo}, {3.00, 34.6, z_trabajo}, {3.0, 30.62, z_trabajo}},
  {{-1.4, 38.33, z_trabajo}, {-1.40, 34.40, z_trabajo}, {-1.23, 30.62, z_trabajo}},
  {{-5.45, 38.0, z_trabajo}, {-5.4, 34.30, z_trabajo}, {-5.4, 30.42, z_trabajo}}
};
bool encoder_inicia_bien = true;
volatile bool interrupt_flag = false;
bool ok = false;

float joint_referencia_calibracion[3] = {0.0, 0.0, 0.0};
float encoder_referencia_calibracion[3] = {-1.32, 95.27, 2.11};

int32_t motor_encoder[4] = {0};
int32_t motor_encoder_prev[4] = {0};

float joint_encoder[3] = {0.0};

float ramal_encoder[3] = {0.0};

float joint_encoder_raw[3] = {0.0};

float joint_encoder_raw_prev[3] = {0.0};

float velocidad_motor[4] = {0.0};

float posicion_error[4] = {0};
float int_posicion_error[4] = {0};

float Kp[4] = {1.5, 1.5, 1.5, 0.0};
float Ki[4] = {0.01, 0.01, 0.01, 0.01};

float relacion_transmicion[3] = {185.0/52.5, 32.0/12.0, 32.0/10.0};

LinearPosition target_position = {0.0, 0.0, 0.0};
AngularPosition target_angle = {0.0, 0.0, 0.0};

Robot my_robot = {
  {0.0, 0.0, 0.0},   // q: q1, q2, q3
  {0.0, 0.0, 0.0},   // w: w1, w2, w3
  {0.0, 0.0, 0.0},   // p: x, y, z
  {0.0, 0.0, 0.0}    // v: v_x, v_y, v_z
};

/**
 * @brief Inicializa el sistema: lee encoders, calcula cinemática directa e inversa hacia home.
 *        Establece encoder_inicia_bien según si los ángulos están dentro de los límites articulares.
 */
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
  computeJointsAngle();
  if(my_robot.q.q1>Q1_MAX||my_robot.q.q1<Q1_MIN||my_robot.q.q2>Q2_MAX||my_robot.q.q2<Q2_MIN||my_robot.q.q3>Q3_MAX||my_robot.q.q3<Q3_MIN){
    encoder_inicia_bien = false;
  }else{
    encoder_inicia_bien = true;
  }
  forwardKinematics();
  target_position = home_position;
  IKResult my_solution = inverseKinematics(target_position);
  ok = my_solution.hasSolution;
  target_angle = my_solution.q;
}

/**
 * @brief Fija target_position a home_position y calcula la cinemática inversa correspondiente.
 */
void goHome(){
  target_position = home_position;
  IKResult my_solution = inverseKinematics(target_position);
  ok = my_solution.hasSolution;
  target_angle = my_solution.q;
}

/**
 * @brief Comprueba si una posición cartesiana está dentro del espacio de trabajo definido.
 * @param workspace Estructura BoxWorkspace con los límites x, y, z.
 * @param p         Posición cartesiana a verificar.
 * @return true si la posición está dentro del espacio de trabajo, false en caso contrario.
 */
bool isInside(const BoxWorkspace &workspace, const LinearPosition &p)
{
    if (p.x < workspace.xMin || p.x > workspace.xMax)
    {
        return false;
    }

    if (p.y < workspace.yMin || p.y > workspace.yMax)
    {
        return false;
    }

    if (p.z < workspace.zMin || p.z > workspace.zMax)
    {
        return false;
    }

    return true;
}

/**
 * @brief Calcula la cinemática inversa para target_position y actualiza target_angle si hay solución.
 *        Imprime el resultado por Serial.
 */
void applyInverseKinematicsToTarget()
{
    IKResult my_solution = inverseKinematics(target_position);

    ok = my_solution.hasSolution;

    if (ok) {
        target_angle = my_solution.q;

        Serial.println("IK OK");
        Serial.print("Target position: X=");
        Serial.print(target_position.x);
        Serial.print(" Y=");
        Serial.print(target_position.y);
        Serial.print(" Z=");
        Serial.println(target_position.z);

        Serial.print("Target angle: q1=");
        Serial.print(target_angle.q1);
        Serial.print(" q2=");
        Serial.print(target_angle.q2);
        Serial.print(" q3=");
        Serial.println(target_angle.q3);
    } else {
        Serial.println("IK ERROR: no hay solucion para esa posicion.");
    }
}

/**
 * @brief Imprime por Serial la lista de comandos disponibles con ejemplos de uso.
 */
void printHelp(void)
{
    Serial.println("===== COMANDOS DISPONIBLES =====");
    Serial.println("x valor     -> cambia target_position.x");
    Serial.println("y valor     -> cambia target_position.y");
    Serial.println("z valor     -> cambia target_position.z");
    Serial.println("go          -> aplica cinematica inversa al target_position");
    Serial.println("p           -> imprime posicion real del efector");
    Serial.println("t           -> imprime target_position actual");
    Serial.println("home        -> va a home_position y aplica IK");
    Serial.println("help        -> muestra esta ayuda");
    Serial.println("Ejemplos:");
    Serial.println("x 10");
    Serial.println("y 30");
    Serial.println("z 12");
    Serial.println("go");
    Serial.println("===============================");
}

/**
 * @brief Interpreta y ejecuta un comando recibido por Serial (x, y, z, go, p, t, home, help).
 * @param input Cadena de texto con el comando y, opcionalmente, su valor numérico.
 */
void processSerialCommand(String input)
{
    input.trim();

    if (input.length() == 0) {
        return;
    }

    int spaceIndex = input.indexOf(' ');

    String cmd;
    String valueStr;

    if (spaceIndex == -1) {
        cmd = input;
        valueStr = "";
    } else {
        cmd = input.substring(0, spaceIndex);
        valueStr = input.substring(spaceIndex + 1);
        valueStr.trim();
    }

    cmd.toLowerCase();

    if (cmd == "x") {
        if (valueStr.length() == 0) {
            Serial.println("ERROR: falta valor para X. Ejemplo: x 10");
            return;
        }

        target_position.x = valueStr.toFloat();
        Serial.print("Nuevo target X = ");
        Serial.println(target_position.x);
        printSetPointPosition();
    }
    else if (cmd == "y") {
        if (valueStr.length() == 0) {
            Serial.println("ERROR: falta valor para Y. Ejemplo: y 30");
            return;
        }

        target_position.y = valueStr.toFloat();
        Serial.print("Nuevo target Y = ");
        Serial.println(target_position.y);
        printSetPointPosition();
    }
    else if (cmd == "z") {
        if (valueStr.length() == 0) {
            Serial.println("ERROR: falta valor para Z. Ejemplo: z 12");
            return;
        }

        target_position.z = valueStr.toFloat();
        Serial.print("Nuevo target Z = ");
        Serial.println(target_position.z);
        printSetPointPosition();
    }
    else if (cmd == "go") {
        applyInverseKinematicsToTarget();
    }
    else if (cmd == "p") {
        printPosition();
    }
    else if (cmd == "t") {
        printSetPointPosition();
    }
    else if (cmd == "home") {
        target_position = home_position;
        Serial.println("Target cargado: home_position");
        applyInverseKinematicsToTarget();
    }
    else if (cmd == "help") {
        printHelp();
    }
    else {
        Serial.print("Comando no reconocido: ");
        Serial.println(cmd);
        Serial.println("Escribe help para ver los comandos.");
    }
}

/** @brief Constructor por defecto. Inicializa el segmento con puntos y tangentes en el origen. */
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

/**
 * @brief Constructor que inicializa el segmento y construye la tabla de longitud de arco.
 * @param p0 Punto inicial.
 * @param p1 Punto final.
 * @param t0 Tangente en p0.
 * @param t1 Tangente en p1.
 */
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

/**
 * @brief Asigna nuevos puntos y tangentes al segmento y reconstruye la tabla de arco.
 * @param p0 Punto inicial.  @param p1 Punto final.
 * @param t0 Tangente en p0. @param t1 Tangente en p1.
 */
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

/**
 * @brief Evalúa la curva cúbica de Hermite en el parámetro u ∈ [0,1].
 * @param u Parámetro normalizado.
 * @return Posición cartesiana interpolada.
 */
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

/**
 * @brief Calcula la derivada (tangente) de la curva en u.
 * @param u Parámetro normalizado.
 * @return Vector tangente en la posición u.
 */
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

/**
 * @brief Calcula la distancia euclídea 3D entre dos posiciones.
 * @param a Posición origen. @param b Posición destino.
 * @return Distancia en las mismas unidades que las coordenadas.
 */
float HermiteSegment::distance3D(LinearPosition a, LinearPosition b) {
  float dx = b.x - a.x;
  float dy = b.y - a.y;
  float dz = b.z - a.z;

  return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Construye la tabla de longitud de arco (u_table/s_table) con HERMITE_ARC_SAMPLES muestras.
 *        Normaliza s_table a [0,1] y actualiza length.
 */
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

/**
 * @brief Devuelve la longitud total del segmento en unidades de espacio (cm).
 * @return Longitud del segmento.
 */
float HermiteSegment::getLength() {
  if (!tableReady) {
    buildArcLengthTable();
  }

  return length;
}

/**
 * @brief Convierte una longitud de arco normalizada s_norm ∈ [0,1] al parámetro u mediante interpolación lineal.
 * @param s_norm Longitud de arco normalizada.
 * @return Parámetro u correspondiente.
 */
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

/**
 * @brief Evalúa la curva a una longitud de arco normalizada s_norm.
 * @param s_norm Longitud de arco normalizada ∈ [0,1].
 * @return Posición cartesiana en ese punto del arco.
 */
LinearPosition HermiteSegment::evaluateByNormalizedArc(float s_norm) {
  float u = getUFromNormalizedArc(s_norm);
  return evaluate(u);
}

/**
 * @brief Evalúa la curva a una distancia absoluta s_local desde el inicio del segmento.
 * @param s_local Distancia recorrida en el segmento (cm).
 * @return Posición cartesiana en ese punto.
 */
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

/** @brief Constructor. Inicializa la trayectoria sin segmentos. */
TrajectoryPath::TrajectoryPath() {
  numSegments = 0;
  totalLength = 0.0f;
  pathReady = false;

  for (int i = 0; i < MAX_TRAJECTORY_SEGMENTS; i++) {
    segmentStart[i] = 0.0f;
    segmentEnd[i] = 0.0f;
  }
}

/** @brief Elimina todos los segmentos y reinicia la longitud total. */
void TrajectoryPath::clear() {
  numSegments = 0;
  totalLength = 0.0f;
  pathReady = false;

  for (int i = 0; i < MAX_TRAJECTORY_SEGMENTS; i++) {
    segmentStart[i] = 0.0f;
    segmentEnd[i] = 0.0f;
  }
}

/**
 * @brief Añade un segmento Hermite al final de la trayectoria.
 * @param segment Segmento a añadir.
 * @return true si se añadió correctamente, false si se alcanzó MAX_TRAJECTORY_SEGMENTS.
 */
bool TrajectoryPath::addSegment(HermiteSegment segment) {
  if (numSegments >= MAX_TRAJECTORY_SEGMENTS) {
    return false;
  }

  segments[numSegments] = segment;
  numSegments++;

  pathReady = false;

  return true;
}

/**
 * @brief Calcula segmentStart[], segmentEnd[] y totalLength a partir de los segmentos añadidos.
 *        Debe llamarse tras añadir todos los segmentos y antes de evaluar la trayectoria.
 */
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

/**
 * @brief Devuelve la longitud total de la trayectoria (suma de todos los segmentos).
 * @return Longitud total en cm.
 */
float TrajectoryPath::getTotalLength() {
  if (!pathReady) {
    build();
  }

  return totalLength;
}

/** @brief Devuelve el número de segmentos añadidos a la trayectoria. */
int TrajectoryPath::getNumSegments() {
  return numSegments;
}

/**
 * @brief Evalúa la trayectoria completa a una distancia global s_global desde el inicio.
 * @param s_global Distancia recorrida en toda la trayectoria (cm).
 * @return Posición cartesiana correspondiente.
 */
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

/**
 * @brief Evalúa la trayectoria a una distancia normalizada s_norm ∈ [0,1].
 * @param s_norm Fracción de la longitud total.
 * @return Posición cartesiana correspondiente.
 */
LinearPosition TrajectoryPath::evaluateByNormalizedDistance(float s_norm) {
  if (!pathReady) {
    build();
  }

  if (s_norm < 0.0f) s_norm = 0.0f;
  if (s_norm > 1.0f) s_norm = 1.0f;

  float s_global = s_norm * totalLength;

  return evaluateByDistance(s_global);
}

/** @brief Constructor por defecto. Inicializa todos los parámetros del perfil a cero. */
SinusoidalSCurveProfile::SinusoidalSCurveProfile() {
  S = 0.0f;
  Vmax = 0.0f;
  Ta = 0.0f;
  Tc = 0.0f;
  Td = 0.0f;
  Ttotal = 0.0f;
}

/**
 * @brief Constructor que configura el perfil directamente.
 * @param totalDistance Distancia total a recorrer (cm).
 * @param maxVelocity   Velocidad máxima (cm/s).
 * @param accelTime     Tiempo de aceleración y desaceleración (s).
 */
SinusoidalSCurveProfile::SinusoidalSCurveProfile(float totalDistance,
                                                 float maxVelocity,
                                                 float accelTime) {
  setProfile(totalDistance, maxVelocity, accelTime);
}

/**
 * @brief Configura los parámetros del perfil S-curve sinusoidal y calcula los tiempos de fase.
 *        Si la trayectoria es demasiado corta para alcanzar Vmax, aplica perfil triangular.
 * @param totalDistance Distancia total (cm). @param maxVelocity Velocidad máxima (cm/s).
 * @param accelTime     Tiempo de aceleración = tiempo de desaceleración (s).
 */
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

/**
 * @brief Evalúa el perfil en el instante t y devuelve posición, velocidad y aceleración.
 * @param t Tiempo transcurrido desde el inicio (s).
 * @return MotionState con s (distancia), v (velocidad) y a (aceleración) en ese instante.
 */
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

/** @brief Devuelve el tiempo total de la maniobra (Ta + Tc + Td) en segundos. */
float SinusoidalSCurveProfile::getTotalTime() {
  return Ttotal;
}

/** @brief Devuelve la distancia total configurada S (cm). */
float SinusoidalSCurveProfile::getTotalDistance() {
  return S;
}

/** @brief Devuelve la velocidad máxima Vmax (cm/s), que puede diferir de la configurada en perfiles cortos. */
float SinusoidalSCurveProfile::getMaxVelocity() {
  return Vmax;
}

/** @brief Devuelve el tiempo de aceleración Ta (s). */
float SinusoidalSCurveProfile::getAccelTime() {
  return Ta;
}

/** @brief Devuelve el tiempo de velocidad constante Tc (s). */
float SinusoidalSCurveProfile::getConstantTime() {
  return Tc;
}

/**
 * @brief Indica si el perfil ha completado la maniobra.
 * @param t Tiempo transcurrido (s).
 * @return true si t >= Ttotal.
 */
bool SinusoidalSCurveProfile::isFinished(float t) {
  return t >= Ttotal;
}
