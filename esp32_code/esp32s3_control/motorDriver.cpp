#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "motorDriver.h"
#include "encoder.h"
#include "kinematics.h"

/**
 * @brief Escribe un array de bytes en un registro del driver de motores por I2C.
 * @param reg  Dirección del registro destino.
 * @param data Puntero al array de bytes a escribir.
 * @param len  Número de bytes a escribir.
 */
void writeRegister(uint8_t reg, uint8_t *data, uint8_t len)
{
  Wire.beginTransmission(ADDRESS);
  Wire.write(reg);

  for (uint8_t i = 0; i < len; i++)
  {
    Wire.write(data[i]);
  }

  Wire.endTransmission();
}

/**
 * @brief Lee un array de bytes desde un registro del driver de motores por I2C.
 * @param reg    Dirección del registro a leer.
 * @param buffer Puntero al buffer donde se almacenan los bytes leídos.
 * @param len    Número de bytes a leer.
 */
void readRegister(uint8_t reg, uint8_t *buffer, uint8_t len)
{
  Wire.beginTransmission(ADDRESS);
  Wire.write(reg);
  Wire.endTransmission();

  Wire.requestFrom(ADDRESS, len);

  for (uint8_t i = 0; i < len && Wire.available(); i++)
  {
    buffer[i] = Wire.read();
  }
}

/**
 * @brief Configura el tipo de motor (JGB37) y la polaridad del encoder en el driver I2C.
 */
void initDriver()
{
  uint8_t motorType = 3;   // JGB37
  uint8_t polarity = 0;    // ajustar si es necesario

  writeRegister(MOTOR_TYPE_ADDR, &motorType, 1);
  delay(5);

  writeRegister(MOTOR_ENCODER_POLARITY_ADDR, &polarity, 1);
  delay(5);

  Serial.println("Driver configurado");
}

/**
 * @brief Envía consignas de velocidad en RPM a los cuatro motores del driver.
 * @param m1 Velocidad motor 1 en RPM.
 * @param m2 Velocidad motor 2 en RPM.
 * @param m3 Velocidad motor 3 en RPM.
 * @param m4 Velocidad motor 4 en RPM.
 */
void setMotorSpeed(float m1, float m2, float m3, float m4)
{
  uint8_t speeds[4];

  speeds[0] = (uint8_t)(m1*COUNTS_PER_REV_OUTPUT*TsDriver/60.0);
  speeds[1] = (uint8_t)(m2*COUNTS_PER_REV_OUTPUT*TsDriver/60.0);
  speeds[2] = (uint8_t)(m3*COUNTS_PER_REV_OUTPUT*TsDriver/60.0);
  speeds[3] = (uint8_t)(m4*COUNTS_PER_REV_OUTPUT*TsDriver/60.0);

  writeRegister(MOTOR_FIXED_SPEED_ADDR, speeds, 4);
}
/**
 * @brief Convierte una posición articular en grados al marco del encoder y activa el control PID.
 *        Aplica límites articulares y la calibración encoder↔articulación antes de enviar.
 * @param targetPos Posición articular objetivo en grados (q1, q2, q3).
 */
void setRobotJointPosition(AngularPosition targetPos){
  float q1 = targetPos.q1;
  float q2 = targetPos.q2;
  float q3 = targetPos.q3;
  q1 = truncate(q1, Q1_MIN, Q1_MAX);
  q1 = toEncoderFrameGeneral(q1,encoder_referencia_calibracion[0],joint_referencia_calibracion[0], false);
  q2 = truncate(q2, Q2_MIN, Q2_MAX);
  q2 = toEncoderFrameGeneral(q2,encoder_referencia_calibracion[1],joint_referencia_calibracion[1], false);
  q3 = truncate(q3, Q3_MIN, Q3_MAX);
  q3 = toEncoderFrameGeneral(q3,encoder_referencia_calibracion[2],joint_referencia_calibracion[2], true);
  setMotorPosition2(q1, q2, q3, 0);
}

/**
 * @brief Controlador PI de posición para los tres ejes del robot.
 *        Calcula la velocidad de consigna con anti-windup y la envía al driver.
 * @param p1 Posición objetivo del eje 1 en grados (marco encoder).
 * @param p2 Posición objetivo del eje 2 en grados (marco encoder).
 * @param p3 Posición objetivo del eje 3 en grados (marco encoder).
 * @param p4 Posición objetivo del eje 4 (no usado, fijado a 0).
 */
void setMotorPosition2(float p1, float p2, float p3, float p4){
  float speed_sp[4];
  int dir_motor[4] = {1, 1, -1, 1};
  posicion_error[0] = p1-joint_encoder[0];
  posicion_error[1] = p2-joint_encoder[1];
  posicion_error[2] = p3-joint_encoder[2];
  for(int i=0;i<3;i++){
    int_posicion_error[i] += posicion_error[i]*Ts;
    // anti-windup simple
    if (int_posicion_error[i] > 10.0) int_posicion_error[i] = 10.0;
    if (int_posicion_error[i] < -10.0) int_posicion_error[i] = -10.0;
    speed_sp[i] = Kp[i]*posicion_error[i] + Ki[i]*int_posicion_error[i];
    if (speed_sp[i] > 110.0) speed_sp[i] = 110.0;
    if (speed_sp[i] < -110.0) speed_sp[i] = -110.0;
    speed_sp[i] = dir_motor[i] * speed_sp[i];
  }
  speed_sp[3] = 0;
  setMotorSpeed(speed_sp[0], speed_sp[1], speed_sp[2], speed_sp[3]);
}
/**
 * @brief Envía valores de PWM directos a los cuatro motores (modo manual, sin control).
 * @param m1 PWM motor 1 (-128 a 127).
 * @param m2 PWM motor 2 (-128 a 127).
 * @param m3 PWM motor 3 (-128 a 127).
 * @param m4 PWM motor 4 (-128 a 127).
 */
void setMotorPWM(int8_t m1, int8_t m2, int8_t m3, int8_t m4)
{
  uint8_t pwm[4];

  pwm[0] = (uint8_t)m1;
  pwm[1] = (uint8_t)m2;
  pwm[2] = (uint8_t)m3;
  pwm[3] = (uint8_t)m4;

  writeRegister(MOTOR_FIXED_PWM_ADDR, pwm, 4);
}

/**
 * @brief Lee los cuatro encoders relativos del driver por I2C y actualiza motor_encoder[].
 */
void readRelativeEncoders(void)
{
  uint8_t buffer[16];

  readRegister(MOTOR_ENCODER_TOTAL_ADDR, buffer, 16);

  for (int i = 0; i < 4; i++)
  {
    motor_encoder[i] =
      (int32_t)buffer[i * 4] |
      ((int32_t)buffer[i * 4 + 1] << 8) |
      ((int32_t)buffer[i * 4 + 2] << 16) |
      ((int32_t)buffer[i * 4 + 3] << 24);
  }
}

/**
 * @brief Inicializa el bus I2C con los pines SDA y SCL definidos en config.h.
 */
void beginWire(void)
{
  Wire.begin(SDA_PIN, SCL_PIN);
}

/**
 * @brief Calcula la velocidad de cada motor en RPM a partir de la variación del encoder relativo.
 *        Debe llamarse cada Ts segundos para que el resultado sea correcto.
 */
void computeVelocity(void)
{
  for (int i = 0; i < 4; i++)
  {
    velocidad_motor[i] = (motor_encoder[i] - motor_encoder_prev[i]) * 60.0 / (COUNTS_PER_REV_OUTPUT * Ts);
    motor_encoder_prev[i] = motor_encoder[i];
  }
}

/**
 * @brief Imprime por Serial los valores de los tres encoders relativos (motor_encoder[]).
 */
void printRelativeEncoder(void){
  Serial.print(" RE1:");
  Serial.print(motor_encoder[0]);
  Serial.print("|| RE2:");
  Serial.print(motor_encoder[1]);
  Serial.print("|| RE3:");
  Serial.println(motor_encoder[2]);
}
