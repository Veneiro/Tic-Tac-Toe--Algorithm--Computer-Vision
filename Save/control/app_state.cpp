#include "app_contracts.h"

bool tableroPendiente = false;
String tableroRecibidoHttp = "";

int modo = 0;
int ganador = 0;
int tablero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
int tableroAnterior[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

String mensaje = "";
bool modoAutomatico = false;
bool juegoEnCurso = false;
bool turnoMaquina = true;

unsigned long proximoParpadeoRobot = 0;
unsigned long finParpadeoRobot = 0;
unsigned long segundoParpadeoRobot = 0;
bool robotParpadeando = false;
bool robotDoblePendiente = false;

int modoEspecialTipo = 0;
