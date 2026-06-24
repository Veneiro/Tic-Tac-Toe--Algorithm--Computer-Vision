#include "app_contracts.h"

namespace
{
  hw_timer_t *timerControl = NULL;
  TaskHandle_t controlTaskHandle = NULL;

  /** @brief ISR del temporizador (segura para IRAM): notifica a controlTask mediante notificación de tarea FreeRTOS en cada tick. */
  void IRAM_ATTR onTimer()
  {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    vTaskNotifyGiveFromISR(controlTaskHandle, &xHigherPriorityTaskWoken);

    if (xHigherPriorityTaskWoken)
    {
      portYIELD_FROM_ISR();
    }
  }

  /** @brief Tarea de control FreeRTOS: espera la notificación del temporizador, lee los encoders, ejecuta el PID
   *  y actualiza la cinemática directa en cada tick.
   *  @param parameter Parámetro de tarea no utilizado. */
  void controlTask(void *parameter)
  {
    while (true)
    {
      ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

      readAbsoluteEncoder();
      computeJointsAngle();
      // Runtime check: si algún joint está >50° fuera de su límite, el encoder ha fallado.
      if (my_robot.q.q1 < Q1_MIN - 50.0f || my_robot.q.q1 > Q1_MAX + 50.0f ||
          my_robot.q.q2 < Q2_MIN - 50.0f || my_robot.q.q2 > Q2_MAX + 50.0f ||
          my_robot.q.q3 < Q3_MIN - 50.0f || my_robot.q.q3 > Q3_MAX + 50.0f)
      {
        encoder_inicia_bien = false;
      }
      if (ok && encoder_inicia_bien)
      {
        setRobotJointPosition(target_angle);
      }
      forwardKinematics();
      interrupt_flag = true;
    }
  }
}

/** @brief Inicializa SPI, el driver I2C, el estado de los encoders, la tarea FreeRTOS de control,
 *  la ISR del temporizador hardware y la pinza. Debe llamarse una vez durante setup(). */
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
  Serial.println("Robot service ready");
}

/** @brief Resetea el objetivo del robot a la posición de inicio llamando a goHome(). */
void resetPosition(){
  goHome();
}

/** @brief Marcador de posición para resetear las posiciones de las piezas; actualmente sin implementación. */
void robotServiceResetPieces()
{
}

/** @brief Busca la primera pieza azul disponible y ejecuta un pickAndPlace hacia la celda indicada del tablero.
 *  @param fila    Fila objetivo del tablero (0–2).
 *  @param columna Columna objetivo del tablero (0–2).
 *  @return Verdadero si se encontró una pieza y se ejecutó el movimiento; falso en caso contrario. */
bool robotServiceMoveToCell(int fila, int columna)
{
  if (fila < 0 || fila > 2 || columna < 0 || columna > 2)
  {
    Serial.println("[ROBOT] Cell out of range");
    return false;
  }

  if (tablero[fila][columna] != 0)
  {
    Serial.println("[ROBOT] Cell already occupied");
    return false;
  }

  for (int i = 0; i < 5; i++)
  {
    if (right_fichas[i] == 2)
    {
      Serial.printf("[ROBOT] Blue piece at right[%d] → row=%d col=%d\n", i, fila, columna);
      pickAndPlace(array_piezas[i], board_grids[fila][columna]);
      return true;
    }
  }

  for (int i = 0; i < 5; i++)
  {
    if (left_fichas[i] == 2)
    {
      Serial.printf("[ROBOT] Blue piece at left[%d] → row=%d col=%d\n", i, fila, columna);
      pickAndPlace(array_piezas_enemigas[i], board_grids[fila][columna]);
      return true;
    }
  }

  Serial.println("[ROBOT] No blue pieces available on either side");
  return false;
}

/** @brief Cuenta y registra el número de cambios de celda entre dos estados del tablero.
 *  @param previousBoard Estado 3x3 del tablero antes del movimiento.
 *  @param currentBoard  Estado 3x3 del tablero después del movimiento. */
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

  Serial.printf("[ROBOT] Board delta applied: %d changes\n", changes);
}
