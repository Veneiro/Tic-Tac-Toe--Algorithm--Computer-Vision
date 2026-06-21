#include "config.h"
#include "motorDriver.h"
#include "encoder.h"
#include "kinematics.h"
#include <Arduino.h>
#include "gripper.h"

// ===================== VARIABLES =====================

hw_timer_t *timer = NULL;
TaskHandle_t controlTaskHandle = NULL;

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
}


// ===================== LOOP =====================

void loop() {
  // Aquí dejas tu interfaz gráfica
  ///*
  pickAndPlace(array_piezas[0], board_grids[1][0]);
  pickAndPlace(array_piezas[1], board_grids[1][1]);
  pickAndPlace(array_piezas[2], board_grids[1][2]);
  pickAndPlace(array_piezas[3], board_grids[2][0]);
  pickAndPlace(array_piezas[4], board_grids[2][1]);
  //*/
  //printSetPointPosition();
  //printPosition();
  //printAbsoluteEncoder();
  //printJoints();
  /*
  openGripperSmooth();
  delay(500);
  closeGripperSmooth();
  delay(500);
  //*
  /*
   if (Serial.available() > 0)
  {
    
      String input = Serial.readStringUntil('\n');
      processSerialCommand(input);
  }
  //*/
}




