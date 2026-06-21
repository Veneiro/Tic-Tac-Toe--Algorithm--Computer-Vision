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
  {-8.0, 27.00, z_trabajo},
  {-4.3, 27.00, z_trabajo},
  {-1.1, 27.50, z_trabajo},
  {2.5, 27.50, z_trabajo},
  {6.3, 27.50, z_trabajo}
};
const LinearPosition board_grids[3][3] = {
  {{2.40, 38.5, z_trabajo}, {2.50, 34.6, z_trabajo}, {2.50, 31.0, z_trabajo}},
  {{-1.3, 38.2, z_trabajo}, {-1.0, 34.3, z_trabajo}, {-1.0, 31.0, z_trabajo}},
  {{-5.2, 38.0, z_trabajo}, {-5.2, 34.0, z_trabajo}, {-5.0, 30.0, z_trabajo}}
};
// ===================== VARIABLES GLOBALES =====================
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
float Ki[4] = {0.01, 0.01, 0.01, 0.0};

float relacion_transmicion[3] = {
  185.0 / 52.5,
  32.0 / 12.0,
  32.0 / 10.0
};

LinearPosition target_position = {0.0, 0.0, 0.0};
AngularPosition target_angle = {0.0, 0.0, 0.0};

Robot my_robot = {
  {0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0}
};

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
