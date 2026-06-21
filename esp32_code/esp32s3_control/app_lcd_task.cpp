#include "app_contracts.h"

SemaphoreHandle_t lcdMutex = NULL;

/** @brief Tarea FreeRTOS en el núcleo 1: llama a dibujarDecoracionTurnoAuto cada 80 ms usando
 *  un intento de bloqueo recursivo no bloqueante del mutex para que el bucle principal nunca se bloquee.
 *  También hace parpadear un aviso "RETIRE LA MANO" en la fila 0 cuando se detecta una mano.
 *  @param param Parámetro de tarea no utilizado. */
static void _lcdAnimTask(void* param) {
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(80));
        if (!modoAutomatico || !juegoEnCurso) continue;
        // No bloqueante: si el loop principal tiene el mutex, saltamos este frame.
        if (xSemaphoreTakeRecursive(lcdMutex, 0) == pdTRUE) {
            // Aviso parpadeante de mano detectada en fila 0 (intercala con el título normal)
            if (manoDetectadaEnTablero) {
                bool parpadeo = (millis() / 450) % 2 == 0;
                lcd.setCursor(0, 0);
                if (parpadeo) lcd.print("!! RETIRE LA MANO !!");
                else          lcd.print("===  [ BOARD ]   ===");
            }
            dibujarDecoracionTurnoAuto();
            xSemaphoreGiveRecursive(lcdMutex);
        }
    }
}

/** @brief Crea el mutex recursivo del LCD y lanza _lcdAnimTask en el núcleo 1 con prioridad 2. */
void lcdTaskInit() {
    // Mutex recursivo: permite que el loop principal haga LCD_LOCK anidados
    // (ej. menú de pausa que a su vez llama actualizarLCD).
    lcdMutex = xSemaphoreCreateRecursiveMutex();
    // Core 1: mismo core donde Wire.begin() fue llamado → evita conflictos I2C cross-core.
    // Prioridad 2: preempta el loop principal (prio 1) durante esperas de motor/HTTP.
    xTaskCreatePinnedToCore(_lcdAnimTask, "lcdAnim", 4096, NULL, 2, NULL, 1);
}
