#pragma once
#include <Arduino.h>

// Pines físicos de los dos buzzers (resistencias 10 y 8 según aa.ino)
#define BUZZER_PIN_1 38
#define BUZZER_PIN_2 40

enum CancionId {
    CANCION_NINGUNA    = -1,
    CANCION_MARIO_KART =  0,   // Mario Kart DS - Waluigi Pinball (BPM 135)
    CANCION_HUMAN_WIN  =  1,   // New Super Mario Bros. Wii - Level Complete (BPM 120)
    CANCION_ROBOT_WIN  =  2,   // New Super Mario Bros. Wii - Game Over (BPM 145)
    CANCION_BOSS_BATTLE =  3,  // Super Mario Bros. 3 - Boss Battle (BPM 90)
    CANCION_RACE_FANFARE = 4, // Super Mario Kart - Race Fanfare (BPM 122)
    CANCION_DRAW         = 5, // Super Mario Bros. 3 - Game Over (BPM 100)
    NUM_CANCIONES
};

// Inicializa los pines LEDC y arranca la tarea FreeRTOS en núcleo 0.
// Llamar una vez en setup().
void buzzerInit();

// Empieza a reproducir una canción. repetir=true → loop infinito.
void buzzerPlay(CancionId id, bool repetir = true);

void buzzerStop();
void buzzerPause();
void buzzerResume();
bool buzzerSonando();
// Devuelve la duración total en ms de una canción (voz más larga).
unsigned long buzzerDuracion(CancionId id);
