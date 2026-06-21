#pragma once

#include <Arduino.h>
#include <WebServer.h>
#include <LiquidCrystal_I2C.h>

#define LCD_ADDRESS 0x27
#define LCD_COLUMNS 20
#define LCD_ROWS 4

extern const char *ssid;
extern const char *password;
extern const char *raspberryPi_IP;
extern const int raspberryPi_PORT;

extern WebServer server;
extern bool tableroPendiente;
extern String tableroRecibidoHttp;

extern const int pinJoyX;
extern const int pinJoyY;
extern const int pinJoyButton;

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

extern byte charAparicion[8];
extern byte charRobot[8];
extern byte charTrofeo[8];
extern byte charJoy[8];
extern byte charEngranaje1[8];
extern byte charEngranaje2[8];
extern byte charBarraInvertida[8];
extern byte charSnakeHead[8];

extern String mensaje;
extern bool modoAutomatico;
extern bool juegoEnCurso;
extern bool turnoMaquina;

extern unsigned long proximoParpadeoRobot;
extern unsigned long finParpadeoRobot;
extern unsigned long segundoParpadeoRobot;
extern bool robotParpadeando;
extern bool robotDoblePendiente;

extern int modoEspecialTipo;

#define MOTOR_ADDRESS 0x34
extern const int PIN_SCLK;
extern const int PIN_MOSI;
extern const int PIN_MISO;

extern const int PIN_CS1;
extern const int PIN_CS2;
extern const int PIN_CS3;

extern const int PIN_GRIPPER;

extern int32_t motor_encoder[4];
extern int32_t motor_encoder_prev[4];
extern float joint_encoder[3];
extern float joint_encoder_raw[3];
extern float joint_encoder_raw_prev[3];
extern float ramal_encoder[3];
extern float velocidad_motor[4];

extern float posicion_error[4];
extern float int_posicion_error[4];
extern float Kp[4];
extern float Ki[4];
extern float relacion_transmicion[3];
extern Robot my_robot;
extern LinearPosition target_position;
extern AngularPosition target_angle;
extern const LinearPosition home_position;
extern const LinearPosition array_piezas[5];
extern const LinearPosition board_grids[3][3];
extern const int pines_CS[3];
extern bool interrupt_flag;
extern bool ok;
extern float joint_referencia_calibracion[3];
extern float encoder_referencia_calibracion[3];

void connectToWiFi();
bool parseBoardToMatrix(const String &input);
void printBoardSerial();
bool sendMatrixToRaspberry();
void procesarEntradaTablero(const String &entrada);
void handleTablero();
void handleRoot();

void esperarLiberacionBoton(int pin);
void setSemaforo(bool rojo, bool amarillo, bool verde);
void cargarCaracteresBase();
void dibujarDecoracionTurnoAuto();
void mostrarPantallaTurnoInicial(bool empiezaMaquina);
void mostrarBienvenida();
void transicionBarrido();
void esperarSeleccionMenu();
void confirmarInicio();
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

