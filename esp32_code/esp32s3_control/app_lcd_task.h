#pragma once
#include <Arduino.h>
#include <freertos/semphr.h>

extern SemaphoreHandle_t lcdMutex;

// Mutex recursivo: el mismo task puede llamar LCD_LOCK() varias veces sin deadlock.
// La tarea de animación usa xSemaphoreTakeRecursive con timeout=0 (no bloqueante).
#define LCD_LOCK()   xSemaphoreTakeRecursive(lcdMutex, portMAX_DELAY)
#define LCD_UNLOCK() xSemaphoreGiveRecursive(lcdMutex)

// Crea el mutex y lanza la tarea de animación. Llamar una vez en setup().
void lcdTaskInit();
