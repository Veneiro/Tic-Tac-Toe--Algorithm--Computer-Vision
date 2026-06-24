#pragma once

#include <Arduino.h>
#include <WebServer.h>
#include <LiquidCrystal_I2C.h>
#include "config.h"
#include "encoder.h"
#include "motorDriver.h"
#include "kinematics.h"
#include "gripper.h"
#include "robot_service.h"
#include "joystick.h"
#include "app_buzzer.h"
#include "app_lcd_task.h"

#define LCD_ADDRESS 0x27
#define LCD_COLUMNS 20
#define LCD_ROWS 4

extern const char *ssid;
extern const char *password;
extern const char *raspberryPi_IP;
extern const int raspberryPi_PORT;
extern const bool reenviarTableroRaspberry;
extern const char *esp32CamURL;

extern WebServer server;
extern bool tableroPendiente;
extern String tableroRecibidoHttp;

extern const int pinJoyX;
extern const int pinJoyY;
extern const int pinJoyButton;
extern Joystick joystick;
extern BoxWorkspace workspace;

extern LiquidCrystal_I2C lcd;

extern const int pinLedRojo;
extern const int pinLedAmarillo;
extern const int pinLedVerde;
extern const int pinSwitch;
extern const int pinStart;
extern const int pinMenu;
extern const int pinSDA;
extern const int pinSCL;
extern int modo;
extern int ganador;
extern int tablero[3][3];
extern int tableroAnterior[3][3];
extern int left_fichas[5];
extern int right_fichas[5];
extern int fuera_rojo;
extern int fuera_azul;

extern byte charAparicion[8];
extern byte charRobot[8];
extern byte charTrofeo[8];
extern byte charJoy[8];
extern byte charEngranaje1[8];
extern byte charEngranaje2[8];
extern byte charBarraInvertida[8];
extern byte charSnakeHead[8];
extern byte charHumanFace[8];
extern byte charHumanBlink[8];
extern byte charFireLight[8];
extern byte charFireMed[8];
extern byte charFireHot[8];

extern String mensaje;
extern bool modoAutomatico;
extern bool juegoEnCurso;
extern bool turnoMaquina;

// Aviso de mano sobre el tablero (cualquier momento del juego)
extern bool manoDetectadaEnTablero;
extern bool bossBattleActivo;

// Estado de verificación del tablero al inicio de partida
extern bool verificarResultadoPendiente;
extern bool verificarListo;
extern bool verificarTableroLimpio;
extern bool verificarManoEnTablero;

extern unsigned long proximoParpadeoRobot;
extern unsigned long finParpadeoRobot;
extern unsigned long segundoParpadeoRobot;
extern bool robotParpadeando;
extern bool robotDoblePendiente;

extern int modoEspecialTipo;

void connectToWiFi();
bool parseBoardToMatrix(const String &input);
bool parsearVector(const String &src, int *arr, int n);
void printBoardSerial();
bool sendMatrixToRaspberry();
bool solicitarCapturaCamara();
bool solicitarVerificacion();
bool procesarEntradaTablero(const String &entrada);
void handleTablero();
void handleRoot();
void handlePedirFoto();
void handleVerificarResultado();

void esperarLiberacionBoton(int pin);
void setSemaforo(bool rojo, bool amarillo, bool verde);
void cargarCaracteresBase();
void dibujarDecoracionTurnoAuto();
void mostrarPantallaTurnoInicial(bool empiezaMaquina);
void mostrarBienvenida();
void transicionBarrido();
void esperarSeleccionMenu();
void confirmarInicio();
void esperarTableroLimpio();
void ejecutarPartida1();
bool abrirMenuPausa();
void manejarFinDeJuego(int ganador);
void leetablero(String entrada);
void animarEntradaTablero();
void mostrarCopaASCII();
void actualizarLCD();
int comprobarGanador();
void vaciarTablero();

void jugarSnake();
void jugarPacman();

