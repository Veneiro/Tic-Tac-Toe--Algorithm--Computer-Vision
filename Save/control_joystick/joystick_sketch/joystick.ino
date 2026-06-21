#include "config.h"
#include "motorDriver.h"
#include "encoder.h"
#include "kinematics.h"
#include <Arduino.h>
#include "gripper.h"
#include "joystick.h"

// ===================== VARIABLES =====================

hw_timer_t *timer = NULL;
TaskHandle_t controlTaskHandle = NULL;

Joystick joystick(
  pinJoyX,
  pinJoyY,
  pinJoyButton,
    0, 2022, 4096,
    0, 1968, 4096
);

const int pinStart = 35;
bool gripperClosed = false;
bool lastStartState = HIGH;

BoxWorkspace workspace = {
    -10.0f, 9.0f,   // xMin, xMax
     24.0f, 40.0f,  // yMin, yMax
      z_trabajo, 16.0f   // zMin, zMax
};
// ===================== INTERRUPCIÓN =====================
void IRAM_ATTR onTimer() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;

  vTaskNotifyGiveFromISR(controlTaskHandle, &xHigherPriorityTaskWoken);

  if (xHigherPriorityTaskWoken) {
    portYIELD_FROM_ISR();
  }
}

// ===================== TAREA DE CONTROL =====================
void controlTask(void *parameter) {
  while (true) {
    // Espera hasta que la ISR del timer la despierte
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // =====================
    // CONTROL CADA 50 ms
    // =====================

    readAbsoluteEncoder();
    if (ok) {
      setRobotJointPosition(target_angle);
    }
    computeJointsAngle();
    forwardKinematics();
    interrupt_flag = true;
  }
}

// ===================== SETUP =====================
void setup() {
  delay(5000);
  Serial.begin(115200);

  beginWire();
  beginSPI();

  initDriver();

  delay(1000);
  
  pinMode(PIN_CS1, OUTPUT);
  digitalWrite(PIN_CS1, HIGH);

  pinMode(PIN_CS2, OUTPUT);
  digitalWrite(PIN_CS2, HIGH);

  pinMode(PIN_CS3, OUTPUT);
  digitalWrite(PIN_CS3, HIGH);

  delay(1000);

  //setZeroAllEncoders();
  //delay(1000);

  inicilizacion();

  // Crear tarea de control
  xTaskCreatePinnedToCore(
    controlTask,          // función
    "ControlTask",        // nombre
    8192,                 // stack
    NULL,                 // parámetro
    3,                    // prioridad
    &controlTaskHandle,   // handle
    1                     // core
  );

  // Timer a 1 MHz
  timer = timerBegin(1000000);

  // Interrupción cada 50 ms
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 50000, true, 0);

  beginGripper();
  Serial.println("Pinza lista");
  openGripperSmooth();
  Serial.println("Sistema listo");

  /***************************************/

  joystick.begin();
  pinMode(pinStart, INPUT_PULLUP);
  /**************************************/
}


// ===================== LOOP =====================

void loop() {
  // Aquí dejas tu interfaz gráfica
  /*
  pickAndPlace(array_piezas[0], board_grids[2][2]);
  pickAndPlace(array_piezas[1], board_grids[0][0]);
  pickAndPlace(array_piezas[2], board_grids[0][1]);
  pickAndPlace(array_piezas[3], board_grids[0][2]);
  pickAndPlace(array_piezas[4], board_grids[1][0]);
  //*/
  //printSetPointPosition();
  //printPosition();
  //printAbsoluteEncoder();
  //printJoints();
  /*
  openGripperSmooth();
  delay(2000);
  closeGripperSmooth();
  delay(2000);
  //*/
  /*
   if (Serial.available() > 0)
  {
    
      String input = Serial.readStringUntil('\n');
      processSerialCommand(input);
  }
  //*/
//*
  joystick.update();

  float vx = joystick.getVx(V_max);
  float vy = joystick.getVy(V_max);
  float vz = joystick.getVz(V_max);

  bool startState = digitalRead(pinStart);

  // Detectar flanco de pulsación: HIGH -> LOW
  if (lastStartState == HIGH && startState == LOW)
  {
      gripperClosed = !gripperClosed;

      if (gripperClosed)
      {
          closeGripperSmooth();
      }
      else
      {
          openGripperSmooth();
      }
  }

  lastStartState = startState;

  interrupt_flag = false;
  while(interrupt_flag == false);

  LinearPosition candidate = target_position;

  candidate.x += vx * Ts;
  candidate.y += vy * Ts;
  candidate.z += vz * Ts;

  if (isInside(workspace, candidate))
  {
      target_position = candidate;
      IKResult my_solution = inverseKinematics(target_position);
      ok = my_solution.hasSolution;
      target_angle = my_solution.q;
  }

  //Serial.print(" | vx: ");Serial.print(vx);
  //Serial.print(" | vy: ");Serial.print(vy);
  //Serial.print(" | vz: ");Serial.println(vz);
  
  Serial.print(" | Tx: ");
  Serial.print(target_position.x);

  Serial.print(" | Ty: ");
  Serial.print(target_position.y);

  Serial.print(" | Tz: ");
  Serial.println(target_position.z);
//*/
}




