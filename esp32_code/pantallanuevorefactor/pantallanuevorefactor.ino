#include "app_contracts.h"
#include <Wire.h>

// Entry point limpio: setup/loop.
void setup()
{
  //****Inicialización Pines Necesarios y LCD****
  Serial.begin(115200);
  randomSeed(micros());

  // Pines Joystick
  pinMode(pinJoyButton, INPUT_PULLUP);
  analogSetAttenuation(ADC_11db); // Rango completo analógico (0-4095)

  // Pines Semáforo
  pinMode(pinLedRojo, OUTPUT);
  pinMode(pinLedAmarillo, OUTPUT);
  pinMode(pinLedVerde, OUTPUT);
  setSemaforo(false, false, false);

  // Configuración LCD (I2C)
  Wire.begin(pinSDA, pinSCL, 400000);
  lcd.begin(LCD_COLUMNS, LCD_ROWS, LCD_ADDRESS);
  lcd.init();
  lcd.backlight();
  lcd.clear();

  // Cargar caracteres personalizados en la LCD
  cargarCaracteresBase();

  // Pines de botones
  pinMode(pinSwitch, INPUT_PULLUP);
  pinMode(pinStart, INPUT_PULLUP);
  pinMode(pinMenu, INPUT_PULLUP);
  //*********************************************

  // ---- Inicialización WiFi y pantalla de arranque ----
  connectToWiFi();
  mostrarBienvenida();

  server.on("/", HTTP_GET, handleRoot);
  server.on("/tablero", HTTP_POST, handleTablero);
  server.begin();

  Serial.println("Servidor HTTP iniciado en /tablero");
  // --------------------------------------------------
}

void loop()
{
  // 1. Fase de menu
  esperarSeleccionMenu();

  if (modoEspecialTipo == 1)
  {
    // EASTER EGG: Snake Mode
    jugarSnake();
    modoEspecialTipo = 0; // Reset al terminar
  }
  else if (modoEspecialTipo == 2)
  {
    // EASTER EGG: Pacman Mode
    jugarPacman();
    modoEspecialTipo = 0; // Reset al terminar
  }
  else 
  {
    //  2. Fase de juego
    ejecutarPartida1();
  }
}


