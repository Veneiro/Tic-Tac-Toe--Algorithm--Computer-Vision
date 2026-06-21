//****************LIBRERIAS*****************

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

//******************************************

// ==========================================
// RED (servidor)
// ==========================================
const char *ssid = "MIM-DPM-GRUPO-3";
const char *password = "mim-dpm-2026";

// const char *ssid = "Galaxy S23 E57C";
// const char *password = "2ppyahxwjx4g7zu";

// const char *ssid = "Livebox6-593F";
// const char *password = "KhCSzCV5DJ4N";

// const char *ssid = "Livebox6-3935";
// const char *password = "k7R2b2TCTfxk";

const char *raspberryPi_IP = "192.168.1.28";
const int raspberryPi_PORT = 5000;

WebServer server(80);

// Variables HTTP adicionales
bool tableroPendiente = false;
String tableroRecibidoHttp = "";

//******************************************

// ==========================================
// CONFIGURACIÓN DEL JOYSTICK
// ==========================================
const int pinJoyX = 1;
const int pinJoyY = 2;
const int pinJoyButton = 3;

// Definicion de variables
#define LCD_ADDRESS 0x27 // direccion de memoria del LCD
#define LCD_COLUMNS 20 // numero de columnas del LCD
#define LCD_ROWS 4 // numero de filas del LCD

// Defino los objetos que voy a utilizar
LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS); // creamos un objeto llamado lcd

// Semaforo de LEDs
const int pinLedRojo = 4;
const int pinLedAmarillo = 5;
const int pinLedVerde = 6;

const int pinSwitch = 36;
const int pinStart = 35;
const int pinMenu = 42;

const int pinSDA = 8;
const int pinSCL = 9;

// Definicion de variables globales
int modo = 0; // 0 = Automático, 1 = Manual
int ganador = 0; // Se inicia sin ganador
int tablero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

// --- Memoria del tablero y animación ---
int tableroAnterior[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

// Animación de aparición de ficha (Destello de luz)
byte charAparicion[8] = {
  0b00000, 0b00100, 0b01010, 0b10101, 0b01010, 0b00100, 0b00000, 0b00000
};
// ----------------------------------------------

// Dibujo de un pequeño robot (8x5 pixeles)
byte charRobot[8] = {
  0b00000, 0b01010, 0b11111, 0b10101, 0b11111, 0b10001, 0b01110, 0b00000
};

// Dibujo de un trofeo
byte charTrofeo[8] = {
  0b11111, 0b10101, 0b10101, 0b01110, 0b00100, 0b00100, 0b01110, 0b00000
};

// Dibujo de un joystick
byte charJoy[8] = {
  0b00100, 0b01110, 0b00100, 0b00100, 0b01110, 0b11111, 0b11111, 0b00000
};

// Animación de un engranaje girando (Fotograma 1 y 2)
byte charEngranaje1[8] = {0b00000, 0b01010, 0b00100, 0b11111, 0b00100, 0b01010, 0b00000, 0b00000};
byte charEngranaje2[8] = {0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000, 0b00000};

// Barra invertida customizada (para evitar problema con caracteres especiales en pantallas japonesas)
byte charBarraInvertida[8] = {
  0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000, 0b00000, 0b00000
};

///////////
String mensaje = "";
bool modoAutomatico = false;
bool juegoEnCurso = false;
bool turnoMaquina = true; // Solo aplica en modo automatico

// Estado de parpadeo del robot lateral en turno de maquina.
unsigned long proximoParpadeoRobot = 0;
unsigned long finParpadeoRobot = 0;
unsigned long segundoParpadeoRobot = 0;
bool robotParpadeando = false;
bool robotDoblePendiente = false;

// --- VARIABLE PARA EL EASTER EGG (NUEVO) ---
int modoEspecialTipo = 0; // 0 = TicTacToe, 1 = Snake, 2 = Pacman

// --- DIBUJO DE CABEZA DE SERPIENTE (NUEVO) ---
// Es un círculo para distinguirlo de los bloques cuadrados del cuerpo
byte charSnakeHead[8] = {0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000};

// -------------------------------------------------------
// PROTOTIPOS ADELANTADOS (funciones del servidor)
// -------------------------------------------------------
void connectToWiFi();
bool parseBoardToMatrix(const String &input);
void printBoardSerial();
bool sendMatrixToRaspberry();
void procesarEntradaTablero(const String &entrada);
void handleTablero();
void handleRoot();
bool abrirMenuPausa();
void manejarFinDeJuego(int ganador);
int comprobarGanador();
void vaciarTablero();
void actualizarLCD();
void jugarSnake(); // Prototipo del Easter Egg
void jugarPacman();
void setSemaforo(bool rojo, bool amarillo, bool verde);
void dibujarDecoracionTurnoAuto();
void mostrarPantallaTurnoInicial(bool empiezaMaquina);
void cargarCaracteresBase();

void esperarLiberacionBoton(int pin)
{
  while (digitalRead(pin) == HIGH)
  {
    server.handleClient();
    delay(10);
  }
  delay(50); // Anti-rebote extra al soltar
}

void setSemaforo(bool rojo, bool amarillo, bool verde)
{
  digitalWrite(pinLedRojo, rojo ? HIGH : LOW);
  digitalWrite(pinLedAmarillo, amarillo ? HIGH : LOW);
  digitalWrite(pinLedVerde, verde ? HIGH : LOW);
}

void cargarCaracteresBase()
{
  lcd.createChar(0, charRobot);
  lcd.createChar(1, charTrofeo);
  lcd.createChar(2, charJoy);
  lcd.createChar(3, charAparicion);
  lcd.createChar(4, charBarraInvertida);
  lcd.createChar(5, charEngranaje1);
  lcd.createChar(6, charEngranaje2);
  lcd.createChar(7, charSnakeHead);
}

void dibujarDecoracionTurnoAuto()
{
  if (!modoAutomatico)
    return;

  // Margen izquierdo: 5 columnas (0..4)
  // Margen derecho: 6 columnas (14..19)
  if (turnoMaquina)
  {
    // Parpadeo no bloqueante y aleatorio, con opcion de doble parpadeo.
    unsigned long ahora = millis();

    if (proximoParpadeoRobot == 0)
    {
      proximoParpadeoRobot = ahora + (unsigned long)random(1400, 4200);
    }

    if (!robotParpadeando && ahora >= proximoParpadeoRobot)
    {
      robotParpadeando = true;
      finParpadeoRobot = ahora + (unsigned long)random(90, 170);
      if (random(100) < 35)
      {
        robotDoblePendiente = true;
        segundoParpadeoRobot = finParpadeoRobot + (unsigned long)random(90, 220);
      }
      else
      {
        robotDoblePendiente = false;
      }
    }

    if (robotParpadeando && ahora >= finParpadeoRobot)
    {
      robotParpadeando = false;
      if (!robotDoblePendiente)
      {
        proximoParpadeoRobot = ahora + (unsigned long)random(1400, 4800);
      }
    }

    if (!robotParpadeando && robotDoblePendiente && ahora >= segundoParpadeoRobot)
    {
      robotParpadeando = true;
      robotDoblePendiente = false;
      finParpadeoRobot = ahora + (unsigned long)random(80, 150);
      proximoParpadeoRobot = finParpadeoRobot + (unsigned long)random(1600, 5200);
    }

    bool ojosCerrados = robotParpadeando;

    // Turno maquina: cabeza de robot a la izquierda
    lcd.setCursor(1, 1); lcd.print("_|_");
    lcd.setCursor(1, 2); lcd.print(ojosCerrados ? ">_<" : "O_O");
    lcd.setCursor(1, 3); lcd.print("|:|");

    // Limpiar monigote derecho
    lcd.setCursor(14, 1); lcd.print("      ");
    lcd.setCursor(14, 2); lcd.print("      ");
    lcd.setCursor(14, 3); lcd.print("      ");
  }
  else
  {
    robotParpadeando = false;
    robotDoblePendiente = false;
    proximoParpadeoRobot = millis() + (unsigned long)random(1200, 3600);

    // Limpiar brazo izquierdo
    lcd.setCursor(0, 1); lcd.print("     ");
    lcd.setCursor(0, 2); lcd.print("     ");
    lcd.setCursor(0, 3); lcd.print("     ");

    // Turno humano: cabeza de persona a la derecha
    lcd.setCursor(15, 1); lcd.print(" o ");
    lcd.setCursor(15, 2); lcd.print("/|"); lcd.write(4);
    lcd.setCursor(15, 3); lcd.print("/ "); lcd.write(4);
  }
}

void mostrarPantallaTurnoInicial(bool empiezaMaquina)
{
  const String lineaTurno = empiezaMaquina ? "Player 2" : "Player 1";
  const int xTurno = (20 - (int)lineaTurno.length()) / 2;

  // Estados: 0 = cayendo, 1..4 = impacto/onda.
  const int gotas = 8;
  byte estado[gotas];
  int gotaX[gotas];
  int gotaY[gotas];
  unsigned long proximoPaso[gotas];

  for (int i = 0; i < gotas; i++)
  {
    estado[i] = 0;
    gotaX[i] = random(0, 20);
    gotaY[i] = random(-8, 1);
    proximoPaso[i] = millis() + (unsigned long)random(40, 180);
  }

  char lastScreen[4][21] = {
    "                    ",
    "                    ",
    "                    ",
    "                    "
  };

  unsigned long inicio = millis();
  unsigned long duracion = 5200;
  unsigned long ultimoTickGotas = 0;

  while (millis() - inicio < duracion)
  {
    server.handleClient();
    unsigned long t = millis() - inicio;

    // Avance temporal de gotas + ondas con tiempos aleatorios.
    if (millis() - ultimoTickGotas >= 25)
    {
      for (int i = 0; i < gotas; i++)
      {
        if (millis() < proximoPaso[i]) continue;

        if (estado[i] == 0)
        {
          gotaY[i]++;
          if (gotaY[i] >= 3)
          {
            gotaY[i] = 3;
            estado[i] = 1;
            proximoPaso[i] = millis() + (unsigned long)random(65, 120);
          }
          else
          {
            proximoPaso[i] = millis() + (unsigned long)random(55, 125);
          }
        }
        else
        {
          estado[i]++;
          if (estado[i] > 4)
          {
            estado[i] = 0;
            gotaX[i] = random(0, 20);
            gotaY[i] = random(-10, -2);
            proximoPaso[i] = millis() + (unsigned long)random(120, 420);
          }
          else
          {
            proximoPaso[i] = millis() + (unsigned long)random(70, 130);
          }
        }
      }
      ultimoTickGotas = millis();
    }

    char screen[4][21];
    for (int r = 0; r < 4; r++) strcpy(screen[r], "                    ");

    // Lluvia ASCII con impacto y ondas (estilo rain-drops).
    for (int i = 0; i < gotas; i++)
    {
      int x = gotaX[i];

      if (estado[i] == 0)
      {
        int y = gotaY[i];
        if (y >= 0 && y < 4 && x >= 0 && x < 20) screen[y][x] = '|';
        if (y - 1 >= 0 && y - 1 < 4 && x >= 0 && x < 20) screen[y - 1][x] = '.';
      }
      else if (estado[i] == 1)
      {
        if (x >= 0 && x < 20) screen[3][x] = 'o';
      }
      else if (estado[i] == 2)
      {
        if (x - 1 >= 0) screen[3][x - 1] = '(';
        if (x + 1 < 20) screen[3][x + 1] = ')';
        if (x >= 0 && x < 20) screen[2][x] = '.';
      }
      else if (estado[i] == 3)
      {
        if (x - 2 >= 0) screen[3][x - 2] = '(';
        if (x + 2 < 20) screen[3][x + 2] = ')';
        if (x - 1 >= 0) screen[3][x - 1] = '-';
        if (x + 1 < 20) screen[3][x + 1] = '-';
        if (x >= 0 && x < 20) screen[3][x] = '_';
      }
      else if (estado[i] == 4)
      {
        if (x - 3 >= 0) screen[3][x - 3] = '.';
        if (x + 3 < 20) screen[3][x + 3] = '.';
        if (x - 2 >= 0) screen[3][x - 2] = '-';
        if (x + 2 < 20) screen[3][x + 2] = '-';
      }
    }

    int letrasTurno = 0;
    if (t > 700)
    {
      letrasTurno = (t - 700) / 90;
      if (letrasTurno > (int)lineaTurno.length()) letrasTurno = lineaTurno.length();
    }

    for (int i = 0; i < letrasTurno; i++)
    {
      int x = xTurno + i;
      if (x >= 0 && x < 20) screen[1][x] = lineaTurno[i];
    }

    // Cursor parpadeante durante la escritura.
    bool showCursor = (t / 220) % 2 == 0;
    if (showCursor)
    {
      if (t > 700 && letrasTurno < (int)lineaTurno.length())
      {
        int cx = xTurno + letrasTurno;
        if (cx >= 0 && cx < 20) screen[1][cx] = (char)255;
      }
    }

    for (int r = 0; r < 4; r++)
    {
      if (memcmp(screen[r], lastScreen[r], 20) != 0)
      {
        lcd.setCursor(0, r);
        for (int c = 0; c < 20; c++)
        {
          if (screen[r][c] == (char)255) lcd.write(255);
          else lcd.print(screen[r][c]);
        }
        memcpy(lastScreen[r], screen[r], 21);
      }
    }

    delay(40);
  }

  transicionBarrido();
  lcd.clear();
}

void setup()
{
  //**********INICIALIZACION*****************
  Serial.begin(115200);
  randomSeed(micros());

  // Pines Joystick
  pinMode(pinJoyButton, INPUT_PULLUP);
  analogSetAttenuation(ADC_11db); // Rango completo analógico (0-4095)

  pinMode(pinLedRojo, OUTPUT);
  pinMode(pinLedAmarillo, OUTPUT);
  pinMode(pinLedVerde, OUTPUT);
  setSemaforo(false, false, false);

  Wire.begin(SDA, SCL, 400000);
  lcd.begin(LCD_COLUMNS, LCD_ROWS, LCD_ADDRESS);
  lcd.init();
  lcd.backlight();
  lcd.clear();

  cargarCaracteresBase();

  pinMode(pinSwitch, INPUT_PULLUP);
  pinMode(pinStart, INPUT_PULLUP);
  pinMode(pinMenu, INPUT_PULLUP);
  //*********************************************

  // ---- Inicialización WiFi y pantalla de arranque ----
  // Iniciamos la conexión primero para que la pantalla CONNECTING...
  // permanezca activa hasta que la WiFi esté realmente conectada.
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

void mostrarBienvenida()
{
  lcd.clear();

  float starX[8] = {10, 10, 10, 10, 9, 9, 9, 9};
  int starY[8]   = {0, 1, 2, 3, 0, 1, 2, 3};
  float speed[8] = {0.4, 0.8, 0.5, 0.9, -0.6, -0.4, -1.0, -0.5};

  String linea1 = "TIC-TAC-TOE";     
  String linea2 = "CONNECTING...";   

  unsigned long startTime = millis();
  unsigned long duracionMinimaAnimacion = 8000;
  unsigned long ultimoPuntoSerial = 0;

  // Mantiene la animación al menos su duración mínima y,
  // si aún no hay WiFi, permanece en CONNECTING... hasta conectar.
  while ((millis() - startTime < duracionMinimaAnimacion) || (WiFi.status() != WL_CONNECTED))
  {
    unsigned long tiempoTranscurrido = millis() - startTime;
    bool wifiConectada = (WiFi.status() == WL_CONNECTED);
    int letrasLinea1 = 0;
    int letrasLinea2 = 0;

    // Fase 1 y 2: Letras de la línea 1
    if (tiempoTranscurrido > 1500) {
      letrasLinea1 = (tiempoTranscurrido - 1500) / 300; 
      if (letrasLinea1 > 11) letrasLinea1 = 11; 
    }
    
    // Fase 3: Letras de la línea 2
    if (tiempoTranscurrido > 5000) {
      letrasLinea2 = (tiempoTranscurrido - 5000) / 200; 
      if (letrasLinea2 > 13) letrasLinea2 = 13;
    }

    // Cuando termina la animación de escritura y la WiFi aún no conecta,
    // dejamos el texto completo fijo en pantalla hasta conectar.
    if (tiempoTranscurrido > duracionMinimaAnimacion && !wifiConectada) {
      letrasLinea1 = 11;
      letrasLinea2 = 13;
    }

    // --- LÓGICA DEL CURSOR BLANCO PARPADEANTE ---
    int cursorX = -1;
    int cursorY = -1;
    bool showCursor = (tiempoTranscurrido / 250) % 2 == 0; // Parpadea cada 250ms

    // Posicionamos el cursor justo donde va a ir la siguiente letra
    if (tiempoTranscurrido > 1500 && tiempoTranscurrido <= 5000) {
      cursorX = 4 + letrasLinea1;
      cursorY = 1;
    } else if (tiempoTranscurrido > 5000) {
      cursorX = 3 + letrasLinea2;
      cursorY = 2;
    }
    // Evitamos que el cursor se salga de la pantalla
    if (cursorX >= 20) cursorX = 19;

    // --- BUCLE DE ESTRELLAS ---
    for (int i = 0; i < 8; i++) 
    {
      int oldX = (int)starX[i];
      bool chocaTexto = false;
      if (starY[i] == 1 && oldX >= 4 && oldX < 4 + letrasLinea1) chocaTexto = true;
      if (starY[i] == 2 && oldX >= 3 && oldX < 3 + letrasLinea2) chocaTexto = true;
      // Proteger también al cursor para que las estrellas no lo borren
      if (showCursor && starY[i] == cursorY && oldX == cursorX) chocaTexto = true;
                         
      if (oldX >= 0 && oldX < 20 && !chocaTexto) {
        lcd.setCursor(oldX, starY[i]); lcd.print(" ");
      }

      starX[i] += speed[i];
      if (starX[i] < 0 || starX[i] >= 20) {
        starX[i] = (speed[i] > 0) ? 10 : 9; 
      }

      int newX = (int)starX[i];
      chocaTexto = false;
      if (starY[i] == 1 && newX >= 4 && newX < 4 + letrasLinea1) chocaTexto = true;
      if (starY[i] == 2 && newX >= 3 && newX < 3 + letrasLinea2) chocaTexto = true;
      if (showCursor && starY[i] == cursorY && newX == cursorX) chocaTexto = true;

      if (newX >= 0 && newX < 20 && !chocaTexto) {
        lcd.setCursor(newX, starY[i]);
        int dist = abs(newX - 9);
        if (dist > 6) lcd.print("*");
        else if (dist > 3) lcd.print("-");
        else lcd.print(".");
      }
    }

    // Dibujar los textos
    if (letrasLinea1 > 0) {
      lcd.setCursor(4, 1); lcd.print(linea1.substring(0, letrasLinea1));
    }
    if (letrasLinea2 > 0) {
      lcd.setCursor(3, 2); lcd.print(linea2.substring(0, letrasLinea2));
    }

    // Dibujar el CURSOR PARPADEANTE 
    if (cursorX != -1 && cursorY != -1) {
      lcd.setCursor(cursorX, cursorY);
      if (showCursor) {
        lcd.write(255); // Bloque sólido
      } else {
        lcd.print(" "); // Lo apaga en su ciclo
      }
    }

    // Retroalimentación en serial mientras espera conexión.
    if (!wifiConectada && millis() - ultimoPuntoSerial >= 500) {
      Serial.print(".");
      ultimoPuntoSerial = millis();
    }

    delay(50); 
  }

  Serial.println("\nWiFi conectado");
  Serial.print("IP del ESP32-S3: ");
  Serial.println(WiFi.localIP());

  transicionBarrido();
  lcd.clear();
}

void transicionBarrido() {
  // Cortina de bloques de izquierda a derecha (si no estaba ya en blanco)
  for (int col = 0; col < 20; col++) {
    for (int row = 0; row < 4; row++) {
      lcd.setCursor(col, row);
      lcd.write(255); 
    }
    delay(15);
  }
  // Limpieza de izquierda a derecha simulando el barrido
  for (int col = 0; col < 20; col++) {
    for (int row = 0; row < 4; row++) {
      lcd.setCursor(col, row);
      lcd.print(" ");
    }
    delay(10);
  }
}

void esperarSeleccionMenu()
{
  bool ultimoEstadoSwitch = !digitalRead(pinSwitch); 
  char ultimaFila3[21] = ""; // Guarda la línea entera para evitar parpadeos

  lcd.clear();

  while (digitalRead(pinStart) == LOW)
  {
    server.handleClient(); 
    unsigned long t = millis();

    // 1. ANIMACIÓN DE TÍTULO (Engranajes girando)
    bool frameEngranaje = (t / 250) % 2 == 0; 
    lcd.setCursor(3, 0);
    lcd.write(frameEngranaje ? 5 : 6); 
    lcd.print(" MAIN  MENU ");
    lcd.write(frameEngranaje ? 6 : 5); 

    // 2. DETECCIÓN DEL MODO
    bool lecturaSwitch = digitalRead(pinSwitch);
    if (lecturaSwitch != ultimoEstadoSwitch)
    {
      if (lecturaSwitch == LOW) { 
        lcd.setCursor(0, 1); lcd.print(" --[    AUTO    ]-- ");
        lcd.setCursor(9, 2); lcd.write(0); 
        modoAutomatico = true;  
      } else { 
        lcd.setCursor(0, 1); lcd.print(" --[   MANUAL   ]-- ");
        lcd.setCursor(9, 2); lcd.write(2); 
        modoAutomatico = false; 
      }
      ultimoEstadoSwitch = lecturaSwitch;
    }

    // 3. ANIMACIÓN DE FLECHAS Y EFECTO "BARRIDO" (WIPE)
    char filaActual3[21] = "                    "; // 20 espacios en blanco
    
    // Flechas moviéndose (Adentro / Afuera)
    bool flechasDentro = (t / 400) % 2 == 0;
    if (flechasDentro) {
      filaActual3[2] = '>'; filaActual3[3] = '>';
      filaActual3[15] = '<'; filaActual3[16] = '<';
    } else {
      filaActual3[1] = '>'; filaActual3[2] = '>';
      filaActual3[16] = '<'; filaActual3[17] = '<';
    }

    // Lógica del Barrido de "PULSE START"
    const char* txt = "PRESS START";
    int longitudTexto = 11;
    int cicloText = t % 3000; // Ciclo total de 3 segundos
    
    int startIdx = 0; // Por dónde empieza a verse el texto
    int endIdx = 0;   // Por dónde termina de verse

    // Fase 1 (Aparece de Izquierda a Derecha)
    if (cicloText <= 1200) {
      startIdx = 0;
      endIdx = (cicloText * longitudTexto) / 1200;
    } 
    // Fase 2 (Se mantiene el texto completo visible)
    else if (cicloText <= 1800) {
      startIdx = 0;
      endIdx = longitudTexto;
    } 
    // Fase 3 (Desaparece de Izquierda a Derecha - Efecto Escoba)
    else {
      startIdx = ((cicloText - 1800) * longitudTexto) / 1200;
      endIdx = longitudTexto;
    }
    
    // Pegamos las letras en la posición central (columna 4)
    // Solo si están dentro del rango visible [startIdx, endIdx)
    for (int i = 0; i < longitudTexto; i++) {
      if (i >= startIdx && i < endIdx) {
        filaActual3[4 + i] = txt[i]; 
      }
    }

    // Imprimimos la fila entera SOLO SI HA CAMBIADO (Sin parpadeos)
    if (strcmp(filaActual3, ultimaFila3) != 0) {
      lcd.setCursor(0, 3);
      lcd.print(filaActual3);
      strcpy(ultimaFila3, filaActual3); 
    }

    delay(20); 
  }
  
  // --- CHEQUEO DE EASTER EGGS ---
  // MENU + JOY -> Pacman
  // MENU solo -> Snake
  if (digitalRead(pinMenu) == HIGH && digitalRead(pinJoyButton) == LOW) {
    modoEspecialTipo = 2;
  } else if (digitalRead(pinMenu) == HIGH) {
    modoEspecialTipo = 1;
  } else {
    modoEspecialTipo = 0;
  }

  esperarLiberacionBoton(pinStart);
  
  if (modoEspecialTipo != 0) {
    esperarLiberacionBoton(pinMenu);
    if (modoEspecialTipo == 2) {
      while (digitalRead(pinJoyButton) == LOW) {
        server.handleClient();
        delay(10);
      }
      delay(40);
    }
  } else {
    confirmarInicio();
  }
}

void confirmarInicio()
{
  lcd.clear();

  if (modoAutomatico) 
  {
    // ==========================================
    // ANIMACIÓN ASCII: ROBOT DESPERTANDO
    // ==========================================
    // Fase 1: Apagado
    lcd.setCursor(0, 0); lcd.print("     ___            ");
    lcd.setCursor(0, 1); lcd.print("    [off]           ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|" ); lcd.write(4); lcd.print("          ");
    lcd.setCursor(0, 3); lcd.print("   ==| |==          ");
    delay(500);
    
    // Fase 2: Antenas arriba y encendiendo
    lcd.setCursor(0, 0); lcd.print("     _|_            ");
    lcd.setCursor(0, 1); lcd.print("    [-_-]           ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4);; lcd.print("          ");
    delay(400);

    // Fase 3: Totalmente encendido con texto
    lcd.setCursor(0, 0); lcd.print("     " ); lcd.write(4); lcd.print("|/            ");
    lcd.setCursor(0, 1); lcd.print("    [O_O]   AUTO    ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4);lcd.print("  MODE    ");
    delay(400);
    
    // Fase 4: Parpadeo de los ojos
    for(int i = 0; i < 3; i++) {
       lcd.setCursor(5, 1); lcd.print(">_<"); delay(150);
       lcd.setCursor(5, 1); lcd.print("O_O"); delay(150);
    }
    delay(400);
  } 
  else 
  {
    // ==========================================
    // ANIMACIÓN: CALIBRACIÓN DE EJES (MODO MANUAL)
    // ==========================================
    lcd.setCursor(3, 0); 
    lcd.print("MANUAL MODE OK");
    
    // Dibujamos el "chasis" de las barras de calibración
    lcd.setCursor(0, 1); lcd.print("X:[             ]");
    lcd.setCursor(0, 2); lcd.print("Y:[             ]");
    lcd.setCursor(1, 3); lcd.print(" Calibrating Axes ");

    // Simulamos que giran el joystick en un círculo 360º
    // Usamos seno y coseno para que se mueva suave de un lado a otro
    for (float t = 0; t <= 6.28; t += 0.35) 
    {
      // Calculamos la posición del cursor (centro en la col 9, amplitud de 6)
      int posX = 9 + 6 * sin(t); 
      int posY = 9 + 6 * cos(t);

      // Dibujamos el movimiento en la barra X
      lcd.setCursor(3, 1);
      for(int i = 3; i <= 15; i++) {
        if (i == posX) lcd.write(255); // El carácter 255 es un bloque cuadrado sólido (█)
        else lcd.print("-");
      }

      // Dibujamos el movimiento en la barra Y
      lcd.setCursor(3, 2);
      for(int i = 3; i <= 15; i++) {
        if (i == posY) lcd.write(255); 
        else lcd.print("-");
      }

      delay(60); 
    }

    // Mensaje final de confirmación
    lcd.setCursor(1, 3); 
    lcd.print("   SYSTEM READY!   ");
    delay(600);
  }
}

void ejecutarPartida1()
{
  vaciarTablero();
  tableroPendiente = false;
  juegoEnCurso = true;
  setSemaforo(false, false, false);

  int ganador = comprobarGanador();
  bool abortarPartida = false;        
  static bool ultimoEstadoMenu = HIGH; 

  if (modoAutomatico == true)
  {
    // Sorteo de inicio: Jugador 1 (humano) o Jugador 2 (brazo robotico).
    turnoMaquina = (random(100) < 50);
    mostrarPantallaTurnoInicial(turnoMaquina);

    if (turnoMaquina) setSemaforo(true, false, false);
    else setSemaforo(false, false, true);

    // --- NUEVO: ANIMACIÓN TIPO ESCÁNER AL ENTRAR AL TABLERO ---
    animarEntradaTablero(); 
    actualizarLCD();
    dibujarDecoracionTurnoAuto();

    bool ultimoEstadoStartAuto = digitalRead(pinStart);

    while (ganador == 0 && !abortarPartida)
    {
      server.handleClient(); 

      // Refrescar animacion lateral durante el turno de la maquina.
      if (turnoMaquina) {
        dibujarDecoracionTurnoAuto();
      }

      bool lecturaMenu = digitalRead(pinMenu);
      if (lecturaMenu == HIGH && ultimoEstadoMenu == LOW)
      {            
        esperarLiberacionBoton(pinMenu); // Esperar que suelte antes de entrar
        abortarPartida = abrirMenuPausa();
        
        // --- LÓGICA DE SALIDA DRAMÁTICA (AUTO) ---
        if (abortarPartida)
        {
          setSemaforo(false, false, false);

          // Animación: Robot cayéndose y desactivándose
          lcd.clear();
          // Fotograma 1: De pie, cansado
          lcd.setCursor(5, 1); lcd.print("[>_<]");
          lcd.setCursor(3, 2); lcd.print("/|:::|"); lcd.write(4);
          delay(400);

          // Fotograma 2: Empezando a caer (bajar una fila, cambiar brazos)
          lcd.clear();
          lcd.setCursor(5, 2); lcd.print("[>_<]");
          lcd.setCursor(3, 3); lcd.write(4); lcd.print("|:::|"); lcd.print("/");
          delay(300);

          // Fotograma 3: En el suelo, totalmente plano y desactivado
          lcd.clear();
          lcd.setCursor(0, 3); // Empezamos desde el borde izquierdo
          lcd.print("   __(x_x)__/-*puff*"); // Corregido con espacios perfectos
          delay(1000);
          
          lcd.setCursor(0, 1); lcd.print(" AUTOMATIC MODE");
          lcd.setCursor(0, 2); lcd.print("   DEACTIVATED");
          delay(1000);
          break; // Salir del while del juego
        }
        
        // --- CORRECCIÓN AL VOLVER DE LA PAUSA ---
        if (!abortarPartida)
        {
          lcd.clear();
          // Ya no ponemos "ESTADO DEL TABLERO", porque actualizarLCD()
          // ahora se encarga de poner su propio título decorado.
          actualizarLCD();
          dibujarDecoracionTurnoAuto();
        }
      }
      ultimoEstadoMenu = lecturaMenu;

      if (tableroPendiente && !abortarPartida)
      {
        if (turnoMaquina)
        {
          String local = tableroRecibidoHttp;
          tableroPendiente = false;
          procesarEntradaTablero(local);
          ganador = comprobarGanador();

          // Si no termina partida, pasa a turno jugador.
          if (ganador == 0)
          {
            turnoMaquina = false;
            setSemaforo(false, false, true); // Turno jugador: verde
            dibujarDecoracionTurnoAuto();
            Serial.println("[TURN] Turno jugador: pulsa START para continuar");
          }
        }
        else
        {
          tableroPendiente = false;
          Serial.println("[TURN] Tablero descartado: no es turno de maquina");
        }
      }

      // Turno jugador: hasta START no vuelve turno maquina.
      if (!turnoMaquina && !abortarPartida && ganador == 0)
      {
        bool estadoStartAuto = digitalRead(pinStart);
        if (estadoStartAuto == HIGH && ultimoEstadoStartAuto == LOW)
        {
          esperarLiberacionBoton(pinStart);
          turnoMaquina = true;
          setSemaforo(true, false, false);
          dibujarDecoracionTurnoAuto();
          Serial.println("[TURN] Turno maquina: esperando nuevo tablero");
        }
        ultimoEstadoStartAuto = estadoStartAuto;
      }
    }

    juegoEnCurso = false; 
    setSemaforo(false, false, false);

    if (abortarPartida) {
      return; 
    }
    manejarFinDeJuego(ganador);
  }
  else
  {
    // ======================================================
    // MODO MANUAL (Con cambio a eje Z)
    // ======================================================
    setSemaforo(false, true, false);

    lcd.clear();
    lcd.setCursor(2, 0);
    lcd.print("-- MANUAL MODE --");
    lcd.setCursor(0, 3);
    lcd.print(" [MENU] for pause ");

    int ultimoX = -1;
    int ultimoY = -1;
    int ultimoZ = -1;
    
    bool controlandoZ = false; // Estado: False = XY, True = Z
    bool antBotonJoy = digitalRead(pinJoyButton);
    bool forzarDibujado = true; // Flag para obligar a repintar la LCD al cambiar de modo

    while (!abortarPartida)
    {
      server.handleClient(); 

      // --- 1. Detección de pausa ---
      bool lecturaMenu = digitalRead(pinMenu);
      if (lecturaMenu == HIGH && ultimoEstadoMenu == LOW)
      {            
        esperarLiberacionBoton(pinMenu); 
        abortarPartida = abrirMenuPausa();
        
        // --- LÓGICA DE SALIDA DRAMÁTICA (MANUAL) CORREGIDA ---
        if (abortarPartida)
        {
          lcd.clear();
          // Centramos el título (22 caracteres -> usamos 20: "! SISTEMA INESTABLE !")
          lcd.setCursor(3, 0); lcd.print("UNSTABLE SYSTEM!");
          
          // Dibujamos las barras base fijas (Columna 0 a 16)
          lcd.setCursor(0, 1); lcd.print("X:[-------------]");
          lcd.setCursor(0, 2); lcd.print("Y:[-------------]");

          // Bucle de agitación (25 frames)
          for (int i = 0; i < 25; i++) {
            // Rango seguro: de la columna 3 a la 15 (dentro de los corchetes)
            int randomX = random(3, 16); 
            int randomY = random(3, 16);

            // 1. Limpiamos solo el interior de las barras para evitar parpadeo total
            lcd.setCursor(3, 1); lcd.print("-------------");
            lcd.setCursor(3, 2); lcd.print("-------------");

            // 2. Dibujamos el bloque de error (ASCII 255) en la nueva posición aleatoria
            lcd.setCursor(randomX, 1); lcd.write(255); 
            lcd.setCursor(randomY, 2); lcd.write(255);
            
            delay(60); 
          }

          // --- ESTADO FINAL DE ERROR ---
          lcd.clear();
          lcd.setCursor(3, 0); lcd.print("CRITICAL ERROR!!");
          lcd.setCursor(0, 1); lcd.print("X:[XXXXXXXXXXXXX]");
          lcd.setCursor(0, 2); lcd.print("Y:[XXXXXXXXXXXXX]");
          lcd.setCursor(1, 3); lcd.print("LEAVING MANUAL MODE");
          
          delay(1500);
          break; // Salir del while del manual
        }
        
        if (!abortarPartida)
        {
          lcd.clear();
          lcd.setCursor(2, 0);
          lcd.print("--- MANUAL MODE ---");
          lcd.setCursor(0, 3);
          lcd.print(" [MENU] for pause ");
          forzarDibujado = true; // Forzar repintado tras volver de la pausa
        }
      }
      ultimoEstadoMenu = lecturaMenu;

      if (abortarPartida) break;

      // --- 2. Detección del botón del Joystick (Cambio de Modo) ---
      bool actBotonJoy = digitalRead(pinJoyButton);
      // Detectamos si lo acabas de pulsar (Flanco de bajada)
      if (actBotonJoy == LOW && antBotonJoy == HIGH) 
      {
        controlandoZ = !controlandoZ; // Alternar entre XY y Z
        forzarDibujado = true;        // Limpiar la línea de valores en la LCD
        delay(50);                    // Anti-rebote mecánico
      }
      antBotonJoy = actBotonJoy;

      // --- 3. Lectura del Joystick ---
      int valorX = analogRead(pinJoyX);
      int valorY = analogRead(pinJoyY);

      int porcentajeX = map(valorX, 0, 4095, 0, 100);
      int porcentajeY = map(valorY, 0, 4095, 0, 100);
      int porcentajeZ = porcentajeY; // Usamos el movimiento Arriba/Abajo para la Z

      // --- 4. Actualizar LCD según el modo ---
      if (!controlandoZ)
      {
        // === MODO X / Y ===
        if (forzarDibujado || abs(porcentajeX - ultimoX) > 1 || abs(porcentajeY - ultimoY) > 1) {
            char bufferXY[21];
            snprintf(bufferXY, sizeof(bufferXY), " X:%3d%%   Y:%3d%%  ", porcentajeX, porcentajeY);
            lcd.setCursor(0, 1);
            lcd.print(bufferXY);
            
            ultimoX = porcentajeX;
            ultimoY = porcentajeY;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Axes: [ X / Y ]    ");
      }
      else
      {
        // === MODO Z ===
        if (forzarDibujado || abs(porcentajeZ - ultimoZ) > 1) {
            char bufferZ[21];
            // Llenamos de espacios la derecha para borrar lo que quedaba de la "Y:"
            snprintf(bufferZ, sizeof(bufferZ), " Z:%3d%%             ", porcentajeZ);
            lcd.setCursor(0, 1);
            lcd.print(bufferZ);
            
            ultimoZ = porcentajeZ;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Axis:  [   Z   ]    ");
      }

      delay(20); 
    }

    juegoEnCurso = false; 
  setSemaforo(false, false, false);

    if (abortarPartida) {
      return; 
    }
  }
}

bool abrirMenuPausa()
{
  // --- 1. Animación de Apertura (Cortina desde el centro hacia afuera) ---
  for (int i = 0; i < 10; i++) {
    for (int r = 0; r < 4; r++) {
      lcd.setCursor(9 - i, r); lcd.write(255);
      lcd.setCursor(10 + i, r); lcd.write(255);
    }
    delay(15);
  }
  for (int i = 0; i < 10; i++) {
    for (int r = 0; r < 4; r++) {
      lcd.setCursor(i, r); lcd.print(" ");
      lcd.setCursor(19 - i, r); lcd.print(" ");
    }
    delay(15);
  }
  
  // --- 2. Textos base bien centrados ---
  lcd.setCursor(4, 0); lcd.print("-[ PAUSED ]-");
  lcd.setCursor(3, 2); lcd.print("START: Exit");
  lcd.setCursor(3, 3); lcd.print("MENU:  Resume");

  bool antStartPausa = digitalRead(pinStart);
  bool antMenuPausa = digitalRead(pinMenu);
  
  delay(200); // Pausa visual de cortesía
  unsigned long startPausa = millis();

  while (true)
  {
    server.handleClient(); 

    // --- 3. Animación en idle (Parpadeo arcade) ---
    unsigned long t = millis() - startPausa;
    bool frame = (t / 300) % 2 == 0; // Cambia cada 300ms
    
    // Animación del título superior
    lcd.setCursor(2, 0);
    if (frame) {
      lcd.print(">>-[ PAUSED ]-<<");
    } else {
      lcd.print("  -[ PAUSED ]-  ");
    }
    
    // Flechas indicadoras en las opciones
    lcd.setCursor(1, 2); lcd.print(frame ? ">" : " ");
    lcd.setCursor(1, 3); lcd.print(frame ? ">" : " ");

    // --- 4. Lectura de botones ---
    bool actStart = digitalRead(pinStart);
    bool actMenu = digitalRead(pinMenu);

    // FLANCO EN START -> SALIR DE LA PARTIDA
    if (actStart == HIGH && antStartPausa == LOW)
    {
      esperarLiberacionBoton(pinStart); 
      transicionBarrido(); // Animación de cierre
      return true; // ABORTAR
    }
    antStartPausa = actStart;

    // FLANCO EN MENU -> CONTINUAR PARTIDA
    if (actMenu == HIGH && antMenuPausa == LOW)
    {
      esperarLiberacionBoton(pinMenu); 
      transicionBarrido(); // Animación de cierre
      return false; // CONTINUAR
    }
    antMenuPausa = actMenu;
    
    delay(20);
  }
}

// Función auxiliar para no repetir los mensajes de ganador
void manejarFinDeJuego(int ganador)
{
  if (ganador >= 1 && ganador <= 3)
  {
    lcd.clear();
    unsigned long startTime = millis();
    unsigned long duration = 12000; 

    // --- VARIABLES DE CONTROL DE FUEGOS ---
    static int fw_x = 10;     // Posición horizontal
    static int fw_y_max = 1;  // Altura de la explosión (Fila 0, 1 o 2)
    
    String linea1 = "MATCH ENDED";
    String linea2 = "";

    if (ganador == 3) {
      linea2 = "ABSOLUTE DRAW";
    } else {
      linea2 = "PLAYER " + String(ganador) + " WIN!";
    }

    char lastScreen[4][21] = {"                    ", "                    ", "                    ", "                    "};

    while (millis() - startTime < duration)
    {
      unsigned long t = millis() - startTime;
      char screen[4][21];
      for(int i = 0; i < 4; i++) strcpy(screen[i], "                    ");

      // --- LÓGICA DE FUEGO ARTIFICIAL (UNO A LA VEZ) ---
      // Ciclo de 1.5 segundos por cada cohete
      int fw_t = t % 1500; 

      // Al inicio de cada cohete, elegimos nueva posición X e Y aleatoria
      if (fw_t < 45) { 
        fw_x = random(2, 18);     // Horizontal: de columna 2 a 17
        fw_y_max = random(0, 3);  // Vertical: puede explotar en fila 0, 1 o 2
      }

      int x = fw_x;
      int y = fw_y_max; // Fila donde ocurre la explosión principal

      // Fases de la animación ajustadas a la altura Y elegida
      if (fw_t < 400) {
        // Fase 1: Cohete subiendo (solo si la explosión es arriba)
        if (y < 3) screen[3][x] = '|'; 
        if (y < 2 && fw_t > 200) screen[2][x] = '|';
      } 
      else if (fw_t < 600) {
        // Fase 2: El punto antes de estallar en su altura Y
        screen[y][x] = '*'; 
      } 
      else if (fw_t < 900) {
        // Fase 3: Explosión principal en la altura Y
        screen[y][x] = '+';
        if(y > 0) screen[y-1][x] = '|';
        if(y < 3) screen[y+1][x] = '|';
        if(x > 0) screen[y][x-1] = '-';
        if(x < 19) screen[y][x+1] = '-';
        // Diagonales de la explosión
        if(y > 0 && x > 0)  screen[y-1][x-1] = 4;  // Carácter personalizado: barra invertida
        if(y > 0 && x < 19) screen[y-1][x+1] = '/';
        if(y < 3 && x > 0)  screen[y+1][x-1] = '/';
        if(y < 3 && x < 19) screen[y+1][x+1] = 4;  // Carácter personalizado: barra invertida
      } 
      else if (fw_t < 1300) {
        // Fase 4: Chispas finales (disipación)
        if(y > 0 && x > 1)  screen[y-1][x-2] = '.';
        if(y < 3 && x < 18) screen[y+1][x+2] = '.';
        if(y < 2 && x > 1)  screen[y+2][x-2] = '.';
        screen[y][x] = '*';
      }

      // --- TEXTO: MÁQUINA DE ESCRIBIR (Sobreescribe el lienzo) ---
      int let1 = 0, let2 = 0;
      if (t > 2500) {
        let1 = (t - 2500) / 120;
        if (let1 > (int)linea1.length()) let1 = linea1.length();
      }
      if (t > 5000) {
        let2 = (t - 5000) / 120;
        if (let2 > (int)linea2.length()) let2 = linea2.length();
      }

      // Dibujar texto (siempre que no haya una chispa justo ahí, el texto manda)
      for (int i = 0; i < let1; i++) screen[1][2 + i] = linea1[i];
      for (int i = 0; i < let2; i++) screen[2][2 + i] = linea2[i];

      // --- CURSOR PARPADEANTE ---
      int cursorX = -1, cursorY = -1;
      bool showCursor = (t / 250) % 2 == 0;
      if (t > 2500 && t <= 5000 && let1 < (int)linea1.length()) {
        cursorX = 2 + let1; cursorY = 1;
      } else if (t > 5000 && let2 < (int)linea2.length()) {
        cursorX = 2 + let2; cursorY = 2;
      }

      if (cursorX >= 0 && cursorX < 20 && cursorY >= 0 && cursorY < 4) {
         if (showCursor) screen[cursorY][cursorX] = (char)255;
         else screen[cursorY][cursorX] = ' ';
      }

      // --- DIBUJADO EFICIENTE ---
      for (int i = 0; i < 4; i++) {
        if (memcmp(screen[i], lastScreen[i], 20) != 0) {
          lcd.setCursor(0, i);
          for (int j = 0; j < 20; j++) {
            if (screen[i][j] == 4) {  // Código del carácter personalizado
              lcd.write(4);            // Escribe el carácter personalizado
            } else {
              lcd.print(screen[i][j]); // Escribe caracteres normales
            }
          }
          memcpy(lastScreen[i], screen[i], 21);
        }
      }
      delay(40); 
    }

    // Mensaje final
    lcd.clear();
    lcd.setCursor(3, 1); lcd.print("MATCH ENDED");
    lcd.setCursor(4, 2); lcd.print("EXITING...");
    delay(2000);
  }
}

void leetablero(String entrada)
{
  int inicio = entrada.indexOf('{');
  int fin = entrada.indexOf('}');

  if (inicio == -1 || fin == -1)
  {
    Serial.println("Error: Formato de string no valido");
    return;
  }

  String contenido = entrada.substring(inicio + 1, fin);
  int fila = 0;
  int col = 0;

  for (int i = 0; i < contenido.length(); i++)
  {
    char c = contenido.charAt(i);

    if (c == ',')
    {
      col++; 
    }
    else if (c == ';')
    {
      fila++;  
      col = 0; 
    }
    else if (c >= '0' && c <= '2')
    {
      if (fila < 3 && col < 3)
      {
        tablero[fila][col] = c - '0';
      }
    }
  }
}

void animarEntradaTablero() 
{
  lcd.clear();
  
  // Fase 1: El título aparece letra a letra (Efecto escáner terminal)
  // Usamos 20 caracteres exactos para centrarlo perfecto
  String titulo = "===  [ BOARD ]   ===";
  lcd.setCursor(0, 0);
  for(int i = 0; i < titulo.length(); i++) {
    lcd.print(titulo[i]);
    delay(30);
  }
  delay(150);

  // Fase 2: Los corchetes exteriores caen fila por fila (Perfectamente centrados)
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(0, i);
    lcd.print("      [     ]       "); 
    delay(120);
  }

  // Fase 3: Las barras separadoras se dibujan como un láser de arriba a abajo
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(8, i); lcd.print("|");
    lcd.setCursor(10, i); lcd.print("|");
    delay(120);
  }
  
  delay(250); // Pequeña pausa de cortesía antes de que la cámara empiece a leer
}

void mostrarCopaASCII() {
  // Limpiamos las 3 filas del tablero antes de dibujar para evitar basura
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(0, i);
    lcd.print("                    ");
  }

  // FILA 1: /(-------)\
  // 5 espacios + / + (-------) + \ + 4 espacios = 20
  lcd.setCursor(0, 1);
  lcd.print("    /(------)");
  lcd.write(4);
  lcd.print("    ");

  // FILA 2: \_) # 1 (_/
  // 6 espacios + \ + _) # 1 (_/ + 3 espacios = 20
  lcd.setCursor(0, 2);
  lcd.print("    ");
  lcd.write(4);
  lcd.print("_) #1 (_/   ");

  // FILA 3: (_____)
  // 6 espacios + (_____) + 7 espacios = 20
  lcd.setCursor(0, 3);
  lcd.print("      (____)       ");

  delay(2000); // Pausa para ver la copa bien
}

void actualizarLCD()
{
  // 1. Comprobamos si hay alguna ficha nueva que animar
  bool hayCambio = false;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0) {
        hayCambio = true;
      }
    }
  }

  // 2. Decoración superior elegante (20 caracteres exactos)
  lcd.setCursor(0, 0);
  lcd.print("===  [ BOARD ]   ===");

  // 3. ANIMACIÓN DE APARICIÓN (Solo se ejecuta si detecta una ficha nueva)
  if (hayCambio) {
    // FASE 1: Un pequeño puntito donde va a aparecer la ficha
    for (int i = 0; i < 3; i++) {
      lcd.setCursor(0, i + 1);
      lcd.print("      ["); // Espacios ajustados para centrar
      for (int j = 0; j < 3; j++) {
        bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
        if (esNueva) lcd.print(".");
        else if (tableroAnterior[i][j] == 0) lcd.print(" ");
        else if (tableroAnterior[i][j] == 1) lcd.print("X");
        else if (tableroAnterior[i][j] == 2) lcd.print("O");
        if (j < 2) lcd.print("|");
      }
      lcd.print("]       "); // Limpia la basura de la derecha
    }
    delay(150); 

    // FASE 2: El destello de luz
    for (int i = 0; i < 3; i++) {
      lcd.setCursor(0, i + 1);
      lcd.print("      [");
      for (int j = 0; j < 3; j++) {
        bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
        if (esNueva) lcd.write(3); // Carácter personalizado de destello
        else if (tableroAnterior[i][j] == 0) lcd.print(" ");
        else if (tableroAnterior[i][j] == 1) lcd.print("X");
        else if (tableroAnterior[i][j] == 2) lcd.print("O");
        if (j < 2) lcd.print("|");
      }
      lcd.print("]       ");
    }
    delay(150); 
  }

  // 3. Dibujo final y estático del tablero
  lcd.setCursor(0, 0);
  lcd.print("===  [ BOARD ]   ===");
  for (int i = 0; i < 3; i++) {
    lcd.setCursor(0, i + 1);
    lcd.print("      [");
    for (int j = 0; j < 3; j++) {
      if (tablero[i][j] == 0) lcd.print(" ");
      else if (tablero[i][j] == 1) lcd.print("X");
      else if (tablero[i][j] == 2) lcd.print("O");
      if (j < 2) lcd.print("|");
      tableroAnterior[i][j] = tablero[i][j];
    }
    lcd.print("]       ");
  }

  // --- LÓGICA DE LA COPA (SOLO SI HAY GANADOR REAL 1 o 2) ---
  int ganador = comprobarGanador(); // <--- CORREGIDO EL NOMBRE AQUÍ
  
  if (ganador == 1 || ganador == 2) {
    delay(1200);        // Pausa para ver la jugada final
    mostrarCopaASCII();   // Dibujamos la copa encima del tablero
    delay(1500);        // Tiempo para disfrutar el trofeo
  } 
  else if (ganador == 3) {
    // Si es empate, quizás solo una pausa corta antes de los fuegos
    delay(1500);
  }
}

int comprobarGanador()
{
  // 1. Comprobar Filas
  for (int i = 0; i < 3; i++)
  {
    if (tablero[i][0] != 0 && tablero[i][0] == tablero[i][1] && tablero[i][1] == tablero[i][2])
    {
      return tablero[i][0];
    }
  }

  // 2. Comprobar Columnas
  for (int i = 0; i < 3; i++)
  {
    if (tablero[0][i] != 0 && tablero[0][i] == tablero[1][i] && tablero[1][i] == tablero[2][i])
    {
      return tablero[0][i];
    }
  }

  // 3. Comprobar Diagonal Principal (\)
  if (tablero[0][0] != 0 && tablero[0][0] == tablero[1][1] && tablero[1][1] == tablero[2][2])
  {
    return tablero[0][0];
  }

  // 4. Comprobar Diagonal Inversa (/)
  if (tablero[0][2] != 0 && tablero[0][2] == tablero[1][1] && tablero[1][1] == tablero[2][0])
  {
    return tablero[0][2];
  }

  // --- NUEVA LÓGICA: Comprobar si está lleno ---
  bool hayEspacioVacio = false;
  for (int f = 0; f < 3; f++)
  {
    for (int c = 0; c < 3; c++)
    {
      if (tablero[f][c] == 0)
      {
        hayEspacioVacio = true; 
        break;
      }
    }
  }

  if (!hayEspacioVacio)
  {
    return 3; // EMPATE
  }

  return 0; // El juego sigue
}

void printBoardSerial()
{
  Serial.println("Matriz parseada:");
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      Serial.print(tablero[i][j]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

void vaciarTablero()
{
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      tablero[i][j] = 0; 
      tableroAnterior[i][j] = 0;
    }
  }
}

// =========================================================
// FUNCIONES DEL SERVIDOR 
// =========================================================

void connectToWiFi()
{
  Serial.print("Conectando a WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
}

bool parseBoardToMatrix(const String &input)
{
  int values[9];
  int count = 0;
  String token = "";

  for (unsigned int i = 0; i < input.length(); i++)
  {
    char c = input[i];
    if ((c >= '0' && c <= '9') || c == '-')
    {
      token += c;
    }
    else if (token.length() > 0)
    {
      if (count >= 9)
        return false;
      values[count++] = token.toInt();
      token = "";
    }
  }

  if (token.length() > 0)
  {
    if (count >= 9)
      return false;
    values[count++] = token.toInt();
  }

  if (count != 9)
    return false;

  int idx = 0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      tablero[i][j] = values[idx++];
    }
  }

  return true;
}

bool sendMatrixToRaspberry()
{
  String matrizJson = "[";
  for (int i = 0; i < 3; i++)
  {
    matrizJson += "[";
    for (int j = 0; j < 3; j++)
    {
      matrizJson += String(tablero[i][j]);
      if (j < 2)
        matrizJson += ",";
    }
    matrizJson += "]";
    if (i < 2)
      matrizJson += ",";
  }
  matrizJson += "]";

  String payload = "{\"matriz\":" + matrizJson + "}";
  String endpoint = "http://" + String(raspberryPi_IP) + ":" + String(raspberryPi_PORT) + "/movimiento";

  HTTPClient http;
  Serial.print("Reenviando a Raspberry: ");
  Serial.println(endpoint);

  if (!http.begin(endpoint))
  {
    Serial.println("Error: No se pudo iniciar conexion HTTP con Raspberry");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST((uint8_t *)payload.c_str(), payload.length());

  if (httpCode > 0)
  {
    Serial.printf("Raspberry HTTP %d\n", httpCode);
    String response = http.getString();
    if (response.length() > 0)
    {
      Serial.print("Respuesta Raspberry: ");
      Serial.println(response);
    }
    http.end();
    return true;
  }

  Serial.printf("Error HTTP hacia Raspberry: %s\n", http.errorToString(httpCode).c_str());
  http.end();
  return false;
}

void procesarEntradaTablero(const String &entrada)
{
  if (!parseBoardToMatrix(entrada))
  {
    Serial.println("Error: formato de tablero invalido");
    return;
  }

  printBoardSerial();
  actualizarLCD();
  sendMatrixToRaspberry();
}

void handleTablero()
{
  if (!server.hasArg("plain"))
  {
    server.send(400, "text/plain", "Body vacio");
    Serial.println("[RX] Peticion sin body");
    return;
  }

  if (!juegoEnCurso || !modoAutomatico)
  {
    server.send(202, "text/plain", "Ignorado: no esta en partida automatica");
    return;
  }

  if (!turnoMaquina)
  {
    server.send(202, "text/plain", "Ignorado: turno jugador");
    return;
  }

  tableroRecibidoHttp = server.arg("plain");
  tableroPendiente = true;

  Serial.println("\n=============================");
  Serial.println("TABLERO RECIBIDO DESDE ESP32-CAM:");
  Serial.println(tableroRecibidoHttp);
  Serial.println("=============================\n");

  server.send(200, "text/plain", "OK - tablero encolado");
}

void handleRoot()
{
  server.send(200, "text/plain", "ESP32-S3 fusion listo. Usa POST /tablero");
}

// =========================================================
// EASTER EGG: MODO PACMAN
// =========================================================
void jugarPacman() {
  lcd.clear();

  // --- Intro estilo Snake pero con PACMAN ---
  unsigned long startAnim = millis();
  String txt1 = "--- PACMAN ---";
  String txt2 = "EASTER EGG FOUND";
  char lastScreen[4][21] = {
    "                    ",
    "                    ",
    "                    ",
    "                    "
  };

  while (millis() - startAnim < 5200) {
    server.handleClient();
    unsigned long t = millis() - startAnim;
    char screen[4][21];
    for (int i = 0; i < 4; i++) strcpy(screen[i], "                    ");

    int frame = (t / 150) % 4;
    char chars1[] = {'|', '/', '-', 4};
    char chars2[] = {'+', 'x', '*', 'o'};

    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 20; c++) {
        int dist = abs(c - 9) + abs(r - 1);
        if (dist == (t / 200) % 12) screen[r][c] = chars1[frame];
        else if (dist == ((t / 200) + 4) % 12) screen[r][c] = chars2[frame];
      }
    }

    int l1 = (t > 1200) ? (t - 1200) / 110 : 0;
    if (l1 > (int)txt1.length()) l1 = txt1.length();
    int l2 = (t > 2800) ? (t - 2800) / 95 : 0;
    if (l2 > (int)txt2.length()) l2 = txt2.length();

    for (int i = 0; i < l1; i++) screen[1][3 + i] = txt1[i];
    for (int i = 0; i < l2; i++) screen[2][2 + i] = txt2[i];

    bool cur = (t / 250) % 2 == 0;
    if (t > 1200 && t < 2800 && l1 < (int)txt1.length() && cur) screen[1][3 + l1] = (char)255;
    if (t > 2800 && l2 < (int)txt2.length() && cur) screen[2][2 + l2] = (char)255;

    for (int i = 0; i < 4; i++) {
      if (strcmp(screen[i], lastScreen[i]) != 0) {
        lcd.setCursor(0, i);
        for (int j = 0; j < 20; j++) {
          if (screen[i][j] == 4) lcd.write(4);
          else if (screen[i][j] == (char)255) lcd.write(255);
          else lcd.print(screen[i][j]);
        }
        strcpy(lastScreen[i], screen[i]);
      }
    }
    delay(40);
  }

  // --- Caracteres personalizados temporales para Pacman ---
  byte charPacRight[8]  = {0b00000, 0b01110, 0b11011, 0b11100, 0b11100, 0b11011, 0b01110, 0b00000};
  byte charPacLeft[8]   = {0b00000, 0b01110, 0b11011, 0b00111, 0b00111, 0b11011, 0b01110, 0b00000};
  byte charPacClosed[8] = {0b00000, 0b01110, 0b11111, 0b11111, 0b11111, 0b11111, 0b01110, 0b00000};
  byte charGhost[8]     = {0b00000, 0b01110, 0b11111, 0b10101, 0b11111, 0b11111, 0b10101, 0b00000};
  byte charGhostFear[8] = {0b00000, 0b01110, 0b11111, 0b10001, 0b11111, 0b10101, 0b11111, 0b00000};
  byte charLife[8]      = {0b00000, 0b01110, 0b11111, 0b11100, 0b11100, 0b11111, 0b01110, 0b00000};
  byte charFruit[8]     = {0b00000, 0b00100, 0b01110, 0b11111, 0b11111, 0b01110, 0b00100, 0b00000};
  byte charEyes[8]      = {0b00000, 0b00000, 0b11011, 0b11011, 0b00000, 0b00000, 0b00000, 0b00000};

  lcd.createChar(0, charPacRight);
  lcd.createChar(1, charPacLeft);
  lcd.createChar(2, charPacClosed);
  lcd.createChar(3, charGhost);
  lcd.createChar(4, charGhostFear);
  lcd.createChar(5, charLife);
  lcd.createChar(6, charFruit);
  lcd.createChar(7, charEyes);

  bool reiniciar = true;
  while (reiniciar) {
    reiniciar = false;

    const int W = 20;
    const int H = 3;
    byte mapa[H][W]; // 0 vacio, 1 pellet, 2 power pellet, 3 pared

    // Mapa más jugable (sin salida bloqueada al spawn).
    const char *plantilla[H] = {
      "....................",
      ".##....##..##....##.",
      "...................."
    };

    const int G = 4;
    int gStartX[G] = {10, 9, 6, 13};
    int gStartY[G] = {1, 1, 1, 1};
    int gx[G], gy[G], gDir[G], gEstado[G];
    // Estado fantasma: 0 normal, 1 frightened, 2 eaten

    int pacX = 1, pacY = 2;
    int dirPac = 1, nextDirPac = 1; // 0 up, 1 right, 2 down, 3 left

    int score = 0;
    int vidas = 3;
    int nivel = 1;
    int comboFantasmas = 200;
    bool salidaForzada = false;
    bool extraVidaDada = false;

    unsigned long miedoHasta = 0;
    unsigned long lastPacMove = 0;
    unsigned long lastGhostMove = 0;
    unsigned long faseStart = millis();
    unsigned long invulnerableHasta = 0;
    unsigned long frutaHasta = 0;
    bool frutaActiva = false;
    int frutaX = 10;
    int frutaY = 1;

    unsigned long ghostReleaseAt[G];
    unsigned long faseCambio = 0;
    int faseModo = 0; // 0 scatter, 1 chase
    int fasePaso = 0;

    auto iniciarNivel = [&]() {
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          mapa[y][x] = (plantilla[y][x] == '#') ? 3 : 1;
        }
      }

      mapa[0][0] = 2;
      mapa[0][19] = 2;
      mapa[2][0] = 2;
      mapa[2][19] = 2;

      pacX = 1;
      pacY = 2;
      dirPac = 1;
      nextDirPac = 1;

      for (int i = 0; i < G; i++) {
        gx[i] = gStartX[i];
        gy[i] = gStartY[i];
        gDir[i] = (i % 2 == 0) ? 1 : 3;
        gEstado[i] = 0;
      }

      // Salida escalonada como en original.
      unsigned long ahora = millis();
      ghostReleaseAt[0] = ahora + 500;
      ghostReleaseAt[1] = ahora + 2200;
      ghostReleaseAt[2] = ahora + 4200;
      ghostReleaseAt[3] = ahora + 6200;

      faseModo = 0;
      fasePaso = 0;
      faseCambio = ahora + 7000;

      comboFantasmas = 200;
      miedoHasta = 0;
      frutaActiva = false;
      frutaHasta = 0;
      invulnerableHasta = ahora + 1800;
      faseStart = ahora;
      lastPacMove = ahora;
      lastGhostMove = ahora;
    };

    auto puedeMover = [&](int x, int y, int dir) {
      int nx = x, ny = y;
      if (dir == 0) ny--;
      if (dir == 1) nx++;
      if (dir == 2) ny++;
      if (dir == 3) nx--;

      if (nx < 0) nx = W - 1;
      if (nx >= W) nx = 0;
      if (ny < 0 || ny >= H) return false;
      return mapa[ny][nx] != 3;
    };

    auto moverPos = [&](int &x, int &y, int dir) {
      if (dir == 0) y--;
      if (dir == 1) x++;
      if (dir == 2) y++;
      if (dir == 3) x--;
      if (x < 0) x = W - 1;
      if (x >= W) x = 0;
    };

    auto pelletsRestantes = [&]() {
      int p = 0;
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          if (mapa[y][x] == 1 || mapa[y][x] == 2) p++;
        }
      }
      return p;
    };

    auto dirOpuesta = [&](int d) {
      return (d + 2) % 4;
    };

    auto distTaxi = [&](int x1, int y1, int x2, int y2) {
      int dx = abs(x1 - x2);
      dx = min(dx, W - dx);
      int dy = abs(y1 - y2);
      return dx + dy;
    };

    auto targetScatterX = [&](int i) {
      if (i == 0) return 19;
      if (i == 1) return 0;
      if (i == 2) return 19;
      return 0;
    };

    auto targetScatterY = [&](int i) {
      if (i == 0) return 0;
      if (i == 1) return 0;
      if (i == 2) return 2;
      return 2;
    };

    auto escogerDirFantasma = [&](int i, bool frightened) {
      int opciones[4];
      int n = 0;
      for (int d = 0; d < 4; d++) {
        if (!puedeMover(gx[i], gy[i], d)) continue;
        if (d == dirOpuesta(gDir[i])) continue; // evita giros bruscos salvo atajo
        opciones[n++] = d;
      }
      if (n == 0) {
        for (int d = 0; d < 4; d++) {
          if (puedeMover(gx[i], gy[i], d)) opciones[n++] = d;
        }
      }
      if (n == 0) return gDir[i];

      if (frightened) {
        return opciones[random(0, n)];
      }

      int tx = pacX;
      int ty = pacY;

      // Comportamientos inspirados en los originales.
      if (gEstado[i] == 2) {
        tx = gStartX[i];
        ty = gStartY[i];
      } else if (faseModo == 0) {
        tx = targetScatterX(i);
        ty = targetScatterY(i);
      } else {
        if (i == 0) {
          // Blinky: directo a Pac-Man.
          tx = pacX;
          ty = pacY;
        } else if (i == 1) {
          // Pinky: 3 celdas por delante.
          tx = pacX;
          ty = pacY;
          for (int k = 0; k < 3; k++) {
            if (dirPac == 0) ty--;
            if (dirPac == 1) tx++;
            if (dirPac == 2) ty++;
            if (dirPac == 3) tx--;
            if (tx < 0) tx = W - 1;
            if (tx >= W) tx = 0;
          }
        } else if (i == 2) {
          // Inky simplificado: vector entre Blinky y Pac-Man adelantado.
          int ax = pacX, ay = pacY;
          if (dirPac == 1) ax = (ax + 2) % W;
          if (dirPac == 3) ax = (ax - 2 + W) % W;
          if (dirPac == 0) ay = max(0, ay - 2);
          if (dirPac == 2) ay = min(H - 1, ay + 2);
          tx = (ax * 2 - gx[0] + W * 2) % W;
          ty = constrain(ay * 2 - gy[0], 0, H - 1);
        } else {
          // Clyde: persigue lejos, se dispersa cerca.
          if (distTaxi(gx[i], gy[i], pacX, pacY) > 6) {
            tx = pacX;
            ty = pacY;
          } else {
            tx = targetScatterX(i);
            ty = targetScatterY(i);
          }
        }
      }

      int bestD = opciones[0];
      int bestV = 9999;
      for (int k = 0; k < n; k++) {
        int d = opciones[k];
        int nx = gx[i], ny = gy[i];
        if (d == 0) ny--;
        if (d == 1) nx++;
        if (d == 2) ny++;
        if (d == 3) nx--;
        if (nx < 0) nx = W - 1;
        if (nx >= W) nx = 0;
        int v = distTaxi(nx, ny, tx, ty);
        if (v < bestV) {
          bestV = v;
          bestD = d;
        }
      }
      return bestD;
    };

    auto comprobarColisiones = [&](bool &muerto) {
      unsigned long ahora = millis();
      bool frightened = ahora < miedoHasta;
      for (int i = 0; i < G; i++) {
        if (gx[i] == pacX && gy[i] == pacY) {
          if (gEstado[i] == 2) continue;
          if (frightened) {
            gEstado[i] = 2; // comido
            score += comboFantasmas;
            comboFantasmas = min(comboFantasmas * 2, 1600);
          } else if (ahora > invulnerableHasta) {
            muerto = true;
            return;
          }
        }
      }
    };

    iniciarNivel();

    bool jugando = true;
    bool antMenuPacman = digitalRead(pinMenu);
    while (jugando) {
      server.handleClient();

      bool lecturaMenuPacman = digitalRead(pinMenu);
      if (lecturaMenuPacman == HIGH && antMenuPacman == LOW) {
        esperarLiberacionBoton(pinMenu);
        bool salirPacman = abrirMenuPausa();
        if (salirPacman) {
          salidaForzada = true;
          break;
        }

        // Al reanudar, evitamos saltos de movimiento por el tiempo transcurrido en pausa.
        unsigned long reanudar = millis();
        lastPacMove = reanudar;
        lastGhostMove = reanudar;
        continue;
      }
      antMenuPacman = lecturaMenuPacman;

      unsigned long ahora = millis();

      // Ciclo scatter/chase clásico simplificado.
      if (ahora > faseCambio) {
        if (fasePaso < 7) {
          faseModo = (faseModo == 0) ? 1 : 0;
          fasePaso++;
          if (faseModo == 0) faseCambio = ahora + 7000;
          else faseCambio = ahora + 20000;
        } else {
          faseModo = 1;
          faseCambio = ahora + 60000;
        }
      }

      int joyX = analogRead(pinJoyX);
      int joyY = analogRead(pinJoyY);
      int difX = abs(joyX - 2048);
      int difY = abs(joyY - 2048);
      if (difX > difY && difX > 800) {
        if (joyX < 1200) nextDirPac = 3;
        else if (joyX > 2800) nextDirPac = 1;
      } else if (difY > difX && difY > 800) {
        if (joyY < 1200) nextDirPac = 0;
        else if (joyY > 2800) nextDirPac = 2;
      }

      int velPac = max(85, 165 - (nivel - 1) * 7);
      int velGhost = max(100, 205 - (nivel - 1) * 8);
      bool frightened = ahora < miedoHasta;
      if (frightened) velGhost += 35;

      // Fruta bonus (dos apariciones por nivel según pellets restantes).
      int pellets = pelletsRestantes();
      if (!frutaActiva && pellets < 30 && frutaHasta == 0) {
        frutaActiva = true;
        frutaHasta = ahora + 6000;
      } else if (!frutaActiva && pellets < 12 && frutaHasta != 0) {
        frutaActiva = true;
        frutaHasta = ahora + 5000;
      }
      if (frutaActiva && ahora > frutaHasta) frutaActiva = false;

      // READY inicial como en clásico.
      if (ahora - faseStart < 1300) {
        lcd.setCursor(0, 0); lcd.print("READY!      PACMAN ");
      }

      if (ahora - lastPacMove >= (unsigned long)velPac && ahora - faseStart >= 1300) {
        lastPacMove = ahora;

        if (puedeMover(pacX, pacY, nextDirPac)) dirPac = nextDirPac;
        if (puedeMover(pacX, pacY, dirPac)) moverPos(pacX, pacY, dirPac);

        if (mapa[pacY][pacX] == 1) {
          mapa[pacY][pacX] = 0;
          score += 10;
        } else if (mapa[pacY][pacX] == 2) {
          mapa[pacY][pacX] = 0;
          score += 50;
          miedoHasta = ahora + max(2600, 6800 - (nivel - 1) * 420);
          comboFantasmas = 200;
          for (int i = 0; i < G; i++) {
            if (gEstado[i] == 0) gEstado[i] = 1;
          }
        }

        if (frutaActiva && pacX == frutaX && pacY == frutaY) {
          score += 100 + nivel * 20;
          frutaActiva = false;
        }

        if (!extraVidaDada && score >= 10000) {
          vidas++;
          extraVidaDada = true;
        }

        bool muerto = false;
        comprobarColisiones(muerto);
        if (muerto) {
          vidas--;
          if (vidas <= 0) {
            jugando = false;
          } else {
            iniciarNivel();
            delay(350);
          }
        }

        if (pellets == 0) {
          nivel++;
          score += 300;
          iniciarNivel();
        }
      }

      if (ahora - lastGhostMove >= (unsigned long)velGhost && ahora - faseStart >= 1300) {
        lastGhostMove = ahora;

        for (int i = 0; i < G; i++) {
          if (ahora < ghostReleaseAt[i]) continue;

          bool fr = (gEstado[i] == 1) && (ahora < miedoHasta);
          int dirElegida = escogerDirFantasma(i, fr);
          gDir[i] = dirElegida;
          moverPos(gx[i], gy[i], gDir[i]);

          if (gEstado[i] == 2 && gx[i] == gStartX[i] && gy[i] == gStartY[i]) {
            gEstado[i] = 0;
          }
        }

        if (ahora > miedoHasta) {
          for (int i = 0; i < G; i++) {
            if (gEstado[i] == 1) gEstado[i] = 0;
          }
        }

        bool muerto = false;
        comprobarColisiones(muerto);
        if (muerto) {
          vidas--;
          if (vidas <= 0) {
            jugando = false;
          } else {
            iniciarNivel();
            delay(350);
          }
        }
      }

      // --- Render ---
      char hud[21];
      snprintf(hud, sizeof(hud), "L%1d LV%02d S%05d", vidas, nivel, score);
      lcd.setCursor(0, 0);
      lcd.print("                    ");
      lcd.setCursor(0, 0);
      lcd.print(hud);
      if (frutaActiva) {
        lcd.setCursor(18, 0);
        lcd.write(6);
      } else {
        lcd.setCursor(18, 0);
        lcd.write(5);
      }

      bool bocaAbierta = ((millis() / 120) % 2 == 0);
      bool parpadeoMiedo = ((millis() / 180) % 2 == 0);
      bool miedoAcabando = (miedoHasta > 0 && miedoHasta - millis() < 1400);

      for (int y = 0; y < H; y++) {
        lcd.setCursor(0, y + 1);
        for (int x = 0; x < W; x++) {
          bool dibujado = false;

          // Fantasmas
          for (int g = 0; g < G; g++) {
            if (gx[g] == x && gy[g] == y) {
              if (gEstado[g] == 2) lcd.write(7);
              else if (gEstado[g] == 1) {
                if (miedoAcabando && parpadeoMiedo) lcd.write(3);
                else lcd.write(4);
              } else lcd.write(3);
              dibujado = true;
              break;
            }
          }

          // Pac-Man
          if (!dibujado && pacX == x && pacY == y) {
            if (!bocaAbierta || dirPac == 0 || dirPac == 2) lcd.write(2);
            else if (dirPac == 1) lcd.write(0);
            else lcd.write(1);
            dibujado = true;
          }

          // Fruta
          if (!dibujado && frutaActiva && x == frutaX && y == frutaY) {
            lcd.write(6);
            dibujado = true;
          }

          if (!dibujado) {
            if (mapa[y][x] == 3) lcd.print('#');
            else if (mapa[y][x] == 2) lcd.print('o');
            else if (mapa[y][x] == 1) lcd.print('.');
            else lcd.print(' ');
          }
        }
      }

      delay(18);
    }

    if (salidaForzada) {
      transicionBarrido();
      break;
    }

    lcd.clear();
    lcd.setCursor(5, 0); lcd.print("PACMAN OVER");
    char b1[21];
    char b2[21];
    snprintf(b1, sizeof(b1), " SCORE: %05d       ", score);
    snprintf(b2, sizeof(b2), " LEVEL: %02d        ", nivel);
    lcd.setCursor(0, 1); lcd.print(b1);
    lcd.setCursor(0, 2); lcd.print(b2);
    lcd.setCursor(0, 3); lcd.print("START Exit MENU Retry");

    while (true) {
      server.handleClient();
      if (digitalRead(pinStart) == HIGH) {
        esperarLiberacionBoton(pinStart);
        reiniciar = false;
        break;
      }
      if (digitalRead(pinMenu) == HIGH) {
        esperarLiberacionBoton(pinMenu);
        reiniciar = true;
        break;
      }
      delay(20);
    }
  }

  cargarCaracteresBase();
  transicionBarrido();
  lcd.clear();
}

// =========================================================
// EASTER EGG: MODO SNAKE (VERSIÓN MEJORADA)
// =========================================================
// =========================================================
// EASTER EGG: MODO SNAKE (ANIMACIONES + LÓGICA ORIGINAL)
// =========================================================
void jugarSnake() {
  lcd.clear();
  unsigned long startAnim = millis();
  String txt1 = "--- SNAKE ---";
  String txt2 = "EASTER EGG FOUND";
  
  // Memoria de pantalla para animación de entrada
  char lastScreen[4][21] = {"                    ", "                    ", "                    ", "                    "};

  // --- 1. ANIMACIÓN DE ENTRADA (Caleidoscopio) ---
  while(millis() - startAnim < 7000) {
    unsigned long t = millis() - startAnim;
    char screen[4][21];
    for(int i=0; i<4; i++) strcpy(screen[i], "                    ");

    int frame = (t / 150) % 4;
    char chars1[] = {'|', '/', '-', 4};
    char chars2[] = {'+', 'x', '*', 'o'};
    
    for(int r = 0; r < 4; r++) {
       for(int c = 0; c < 20; c++) {
           int dist = abs(c - 9) + abs(r - 1);
           if (dist == (t / 200) % 12) screen[r][c] = chars1[frame];
           else if (dist == ((t / 200) + 4) % 12) screen[r][c] = chars2[frame];
       }
    }

    int l1 = (t > 2500) ? (t - 2500) / 120 : 0;
    if (l1 > (int)txt1.length()) l1 = txt1.length();
    int l2 = (t > 4500) ? (t - 4500) / 120 : 0;
    if (l2 > (int)txt2.length()) l2 = txt2.length();

    for(int i=0; i<l1; i++) screen[1][3+i] = txt1[i];
    for(int i=0; i<l2; i++) screen[2][2+i] = txt2[i];

    bool cur = (t / 250) % 2 == 0;
    if (t > 2500 && t < 4500 && l1 < (int)txt1.length() && cur) screen[1][3+l1] = (char)255;
    if (t > 4500 && l2 < (int)txt2.length() && cur) screen[2][2+l2] = (char)255;

    for(int i=0; i<4; i++) {
      if (strcmp(screen[i], lastScreen[i]) != 0) {
        lcd.setCursor(0, i);
        for (int j = 0; j < 20; j++) {
          if (screen[i][j] == 4) {
            lcd.write(4);
          } else if (screen[i][j] == (char)255) {
            lcd.write(255);
          } else {
            lcd.print(screen[i][j]);
          }
        }
        strcpy(lastScreen[i], screen[i]);
      }
    }
    delay(40);
  }
  lcd.clear();

  // --- Sprites temporales SOLO para Snake ---
  byte snakeHeadRight[8] = {0b00000, 0b01100, 0b11110, 0b11111, 0b11110, 0b01100, 0b00000, 0b00000};
  byte snakeHeadLeft[8]  = {0b00000, 0b00110, 0b01111, 0b11111, 0b01111, 0b00110, 0b00000, 0b00000};
  byte snakeHeadUp[8]    = {0b00000, 0b00100, 0b01110, 0b11111, 0b10101, 0b00100, 0b00000, 0b00000};
  byte snakeHeadDown[8]  = {0b00000, 0b00100, 0b10101, 0b11111, 0b01110, 0b00100, 0b00000, 0b00000};
  byte snakeBody[8]      = {0b00000, 0b01110, 0b11111, 0b11111, 0b11111, 0b01110, 0b00000, 0b00000};
  byte snakeApple[8]     = {0b00100, 0b01010, 0b01110, 0b11111, 0b11111, 0b01110, 0b00100, 0b00000};

  lcd.createChar(0, snakeHeadRight);
  lcd.createChar(1, snakeHeadLeft);
  lcd.createChar(2, snakeHeadUp);
  lcd.createChar(3, snakeHeadDown);
  lcd.createChar(4, snakeBody);
  lcd.createChar(5, snakeApple);

  // --- BUCLE DE REINTENTO (Para no salir al menú principal si mueres) ---
  bool reiniciar = true;

  while (reiniciar) {
    reiniciar = false; // Por defecto no reinicia a menos que pulses MENU
    
    // --- LÓGICA Y VARIABLES ORIGINALES DEL JUEGO ---
    int snakeX[80], snakeY[80];
    int snakeLen = 3;
    int dir = 1; // 0=Arriba, 1=Derecha, 2=Abajo, 3=Izquierda

    // Posición inicial
    for(int i = 0; i < snakeLen; i++) {
      snakeX[i] = 10 - i;
      snakeY[i] = 2;
    }
    
    // Comida inicial
    int foodX = random(0, 20);
    int foodY = random(0, 4);

    bool playing = true;
    int score = 0;
    bool salidaForzada = false;
    bool antMenuSnake = digitalRead(pinMenu);

    auto dibujarSnakeCompleta = [&]() {
      lcd.clear();
      lcd.setCursor(foodX, foodY);
      lcd.write(5);

      for (int i = snakeLen - 1; i > 0; i--) {
        lcd.setCursor(snakeX[i], snakeY[i]);
        lcd.write(4);
      }

      byte headChar = 0;
      if (dir == 0) headChar = 2;
      else if (dir == 1) headChar = 0;
      else if (dir == 2) headChar = 3;
      else if (dir == 3) headChar = 1;
      lcd.setCursor(snakeX[0], snakeY[0]);
      lcd.write(headChar);
    };

    // Pintar estado inicial
    dibujarSnakeCompleta();

    // --- BUCLE PRINCIPAL DE LA PARTIDA ---
    while(playing) {
      server.handleClient();

      // Pausa del Snake con MENU: START sale, MENU reanuda.
      bool lecturaMenuSnake = digitalRead(pinMenu);
      if (lecturaMenuSnake == HIGH && antMenuSnake == LOW) {
        esperarLiberacionBoton(pinMenu);
        bool salirSnake = abrirMenuPausa();
        if (salirSnake) {
          salidaForzada = true;
          break;
        }
        dibujarSnakeCompleta();
      }
      antMenuSnake = lecturaMenuSnake;

      // Espera Activa del Joystick
      unsigned long tiempoFrame = millis();
      while (millis() - tiempoFrame < 250) { 
        server.handleClient(); 
        
        int joyX = analogRead(pinJoyX);
        int joyY = analogRead(pinJoyY);
        int difX = abs(joyX - 2048);
        int difY = abs(joyY - 2048);

        if (difX > difY && difX > 800) {
          if (joyX < 1200 && dir != 1) dir = 3;      
          else if (joyX > 2800 && dir != 3) dir = 1; 
        } 
        else if (difY > difX && difY > 800) {
          if (joyY < 1200 && dir != 2) dir = 0;      
          else if (joyY > 2800 && dir != 0) dir = 2; 
        }
        delay(5); 
      }

      // Borrar la cola antigua de la pantalla
      lcd.setCursor(snakeX[snakeLen-1], snakeY[snakeLen-1]);
      lcd.print(" ");

      int oldHeadX = snakeX[0];
      int oldHeadY = snakeY[0];

      // Desplazar las coordenadas del cuerpo
      for(int i = snakeLen - 1; i > 0; i--) {
        snakeX[i] = snakeX[i-1];
        snakeY[i] = snakeY[i-1];
      }

      // Actualizar la cabeza
      if(dir == 0) snakeY[0]--;
      else if(dir == 1) snakeX[0]++;
      else if(dir == 2) snakeY[0]++;
      else if(dir == 3) snakeX[0]--;

      // Paredes Pac-Man
      if(snakeX[0] < 0) snakeX[0] = 19;
      else if(snakeX[0] >= 20) snakeX[0] = 0;
      if(snakeY[0] < 0) snakeY[0] = 3;
      else if(snakeY[0] >= 4) snakeY[0] = 0;

      // Autocolisión
      for(int i = 1; i < snakeLen; i++) {
        if(snakeX[0] == snakeX[i] && snakeY[0] == snakeY[i]) playing = false;
      }
      
      if(!playing) break;

      // Comer
      if(snakeX[0] == foodX && snakeY[0] == foodY) {
        if(snakeLen < 80) snakeLen++;
        score += 10;
        
        // Generar nueva comida
        bool valid = false;
        while(!valid) {
          foodX = random(0, 20);
          foodY = random(0, 4);
          valid = true;
          for(int i = 0; i < snakeLen; i++) {
            if(foodX == snakeX[i] && foodY == snakeY[i]) valid = false;
          }
        }
      }

      lcd.setCursor(foodX, foodY); lcd.write(5);
      lcd.setCursor(oldHeadX, oldHeadY); lcd.write(4);

      byte headChar = 0;
      if (dir == 0) headChar = 2;
      else if (dir == 1) headChar = 0;
      else if (dir == 2) headChar = 3;
      else if (dir == 3) headChar = 1;
      lcd.setCursor(snakeX[0], snakeY[0]); lcd.write(headChar);
    }

    if (salidaForzada) break; // Si abortaste, salimos del bucle de reintento

    // --- GAME OVER: ONDA EXPANSIVA ---
    delay(500); // Congelar un momento

    int hX = snakeX[0];
    int hY = snakeY[0];
    
    // Calcular expansión en base a la distancia desde la cabeza
    for (int radius = 0; radius <= 22; radius++) {
      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 20; c++) {
          int dist = abs(c - hX) + abs(r - hY);
          if (dist == radius) {
            lcd.setCursor(c, r);
            lcd.write(255); // Bloque sólido
          }
        }
      }
      delay(40); // Velocidad de la onda
    }
    
    delay(200);
    transicionBarrido(); // El barrido limpia la pantalla blanca

    // --- PANTALLA DE PUNTUACIÓN Y MENÚ ---
    lcd.setCursor(5, 0); lcd.print("GAME OVER");
    
    char bufScore[21];
    snprintf(bufScore, sizeof(bufScore), "   SCORE: %04d   ", score);
    lcd.setCursor(0, 1); lcd.print(bufScore);

    lcd.setCursor(0, 2); lcd.print(" START: Exit");
    lcd.setCursor(0, 3); lcd.print(" MENU:  Retry");

    // --- ESPERAR DECISIÓN ---
    while (true) {
      server.handleClient();
      
      // Flechas parpadeantes
      bool blink = (millis() / 300) % 2 == 0;
      lcd.setCursor(15, 2); lcd.print(blink ? "<" : " ");
      lcd.setCursor(15, 3); lcd.print(blink ? "<" : " ");

      // Corregido a HIGH (pulsado)
      if (digitalRead(pinStart) == HIGH) {
        esperarLiberacionBoton(pinStart);
        
        // Animación de SALIDA (Implosión)
        for(int i = 0; i < 10; i++) {
          for(int r = 0; r < 4; r++) {
            lcd.setCursor(i, r); lcd.write(255);
            lcd.setCursor(19 - i, r); lcd.write(255);
          }
          delay(30);
        }
        delay(200);
        transicionBarrido();
        reiniciar = false;
        break;
      }
      
      // Corregido a HIGH (pulsado)
      if (digitalRead(pinMenu) == HIGH) {
        esperarLiberacionBoton(pinMenu);
        
        // Animación de REINTENTO (Cuenta atrás)
        lcd.clear();
        for(int i = 3; i > 0; i--) {
          lcd.setCursor(6, 1); lcd.print("READY: "); lcd.print(i);
          delay(600);
        }
        lcd.clear();
        reiniciar = true;
        break;
      }
      delay(20);
    }
  }

  // Restaurar caracteres del resto del sistema al salir del minijuego.
  cargarCaracteresBase();
}