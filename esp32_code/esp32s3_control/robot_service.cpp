#include "app_contracts.h"

namespace
{
  hw_timer_t *timerControl = NULL;
  TaskHandle_t controlTaskHandle = NULL;

  void IRAM_ATTR onTimer()
  {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    vTaskNotifyGiveFromISR(controlTaskHandle, &xHigherPriorityTaskWoken);

    if (xHigherPriorityTaskWoken)
    {
      portYIELD_FROM_ISR();
    }
  }

  void controlTask(void *parameter)
  {
    while (true)
    {
      ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

      readAbsoluteEncoder();
      if (ok && encoder_inicia_bien)
      {
        setRobotJointPosition(target_angle);
      }
      computeJointsAngle();
      forwardKinematics();
      interrupt_flag = true;
    }
  }
}

void robotServiceInit()
{
  beginSPI();
  initDriver();

  pinMode(PIN_CS1, OUTPUT);
  digitalWrite(PIN_CS1, HIGH);

  pinMode(PIN_CS2, OUTPUT);
  digitalWrite(PIN_CS2, HIGH);

  pinMode(PIN_CS3, OUTPUT);
  digitalWrite(PIN_CS3, HIGH);

  delay(1000);
  inicilizacion();

  xTaskCreatePinnedToCore(
      controlTask,
      "ControlTask",
      8192,
      NULL,
      3,
      &controlTaskHandle,
      1);

  timerControl = timerBegin(1000000); 
  timerAttachInterrupt(timerControl, &onTimer);
  timerAlarm(timerControl, 50000, true, 0);

  beginGripper();
  openGripperSmooth();
  Serial.println("Robot service listo");
}

void resetPosition(){
  goHome();
}

void robotServiceResetPieces()
{
}

bool robotServiceMoveToCell(int fila, int columna)
{
  if (fila < 0 || fila > 2 || columna < 0 || columna > 2)
  {
    Serial.println("[ROBOT] Casilla fuera de rango");
    return false;
  }

  if (tablero[fila][columna] != 0)
  {
    Serial.println("[ROBOT] La casilla indicada ya esta ocupada");
    return false;
  }

  for (int i = 0; i < 5; i++)
  {
    if (right_fichas[i] == 2)
    {
      Serial.printf("[ROBOT] Pieza azul en right[%d] → fila=%d col=%d\n", i, fila, columna);
      pickAndPlace(array_piezas[i], board_grids[fila][columna]);
      return true;
    }
  }

  for (int i = 0; i < 5; i++)
  {
    if (left_fichas[i] == 2)
    {
      Serial.printf("[ROBOT] Pieza azul en left[%d] → fila=%d col=%d\n", i, fila, columna);
      pickAndPlace(array_piezas_enemigas[i], board_grids[fila][columna]);
      return true;
    }
  }

  Serial.println("[ROBOT] Sin piezas azules disponibles en ningun lateral");
  return false;
}

void robotServiceApplyBoardDelta(int previousBoard[3][3], int currentBoard[3][3])
{
  int changes = 0;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (previousBoard[i][j] != currentBoard[i][j])
      {
        changes++;
      }
    }
  }

  Serial.printf("[ROBOT] Board delta aplicado: %d cambios\n", changes);
}
