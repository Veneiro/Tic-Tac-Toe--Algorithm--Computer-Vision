#include "app_contracts.h"

/** @brief Espera activa hasta que el pin del botón especificado pase a LOW (soltado con INPUT_PULLUP),
 *  luego espera 50 ms adicionales para el antirrebote.
 *  @param pin Número de pin GPIO del botón que se espera. */
void esperarLiberacionBoton(int pin)
{
  while (digitalRead(pin) == HIGH)
  {
    server.handleClient();
    delay(10);
  }
  delay(50); // Anti-rebote extra al soltar
}

/** @brief Establece los tres LEDs del semáforo.
 *  @param rojo    Verdadero para encender el LED rojo.
 *  @param amarillo Verdadero para encender el LED amarillo.
 *  @param verde   Verdadero para encender el LED verde. */
void setSemaforo(bool rojo, bool amarillo, bool verde)
{
  digitalWrite(pinLedRojo, rojo ? HIGH : LOW);
  digitalWrite(pinLedAmarillo, amarillo ? HIGH : LOW);
  digitalWrite(pinLedVerde, verde ? HIGH : LOW);
}

/** @brief Carga los 8 caracteres personalizados base del LCD: robot, trofeo, joystick, destello,
 *  barra invertida, engranaje1, engranaje2 y cabeza de serpiente en las ranuras CGRAM 0–7. */
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

// ── Animación de fuego ────────────────────────────────────────────────────────
// Ocupa cols 0-4 (izquierda) y cols 15-19 (derecha) en filas 1-3.
// Usa slots 5-7 redefinidos como llamas mientras está activa.
// calor[col]: 1=chispa, 2=media, 3=intensa. Se actualizan ~cada 130ms.

/** @brief Reemplaza las ranuras CGRAM 5–7 del LCD con caracteres de animación de fuego (llama suave, media e intensa). */
static void _cargarCharsFuego()
{
  lcd.createChar(5, charFireLight);
  lcd.createChar(6, charFireMed);
  lcd.createChar(7, charFireHot);
}

/** @brief Escribe un carácter de celda de fuego en el LCD en la posición actual del cursor.
 *  La intensidad del carácter depende del nivel de calor y la fila vertical (la inferior es siempre intensa).
 *  @param calor Nivel de calor: 1 = chispa, 2 = llama media, 3 = llama intensa.
 *  @param row   Fila del LCD (1–3); la fila 3 siempre se dibuja con intensidad máxima. */
static void _escribirCeldaFuego(uint8_t calor, int row)
{
  if (row == 3)
  {
    lcd.write((uint8_t)7);                // base: siempre intensa
  }
  else if (row == 2)
  {
    if (calor >= 2) lcd.write((uint8_t)6);
    else if (calor == 1) lcd.write((uint8_t)5);
    else lcd.print(' ');
  }
  else  // row == 1
  {
    if (calor == 3) lcd.write((uint8_t)5);
    else lcd.print(' ');
  }
}

/** @brief Dibuja bordes de fuego animados en el LCD (cols 0–4 izquierda, cols 15–19 derecha, filas 1–3).
 *  Los valores de calor se aleatorizan aproximadamente cada 130 ms para crear un efecto de parpadeo. */
void dibujarFuego()
{
  const int COLS_F = 5;
  static unsigned long _tActualizar = 0;
  static int8_t _calorIzq[COLS_F] = {3, 3, 3, 3, 3};
  static int8_t _calorDer[COLS_F] = {3, 3, 3, 3, 3};

  // Actualizar calor ~130 ms
  if (millis() - _tActualizar >= 130)
  {
    for (int c = 0; c < COLS_F; c++)
    {
      _calorIzq[c] = constrain(_calorIzq[c] + (int8_t)random(-1, 2), 1, 3);
      _calorDer[c] = constrain(_calorDer[c] + (int8_t)random(-1, 2), 1, 3);
    }
    _tActualizar = millis();
  }

  // Dibujar margen izquierdo (cols 0-4)
  for (int r = 1; r <= 3; r++)
  {
    lcd.setCursor(0, r);
    for (int c = 0; c < COLS_F; c++)
      _escribirCeldaFuego(_calorIzq[c], r);
  }

  // Dibujar margen derecho (cols 15-19)
  for (int r = 1; r <= 3; r++)
  {
    lcd.setCursor(15, r);
    for (int c = 0; c < COLS_F; c++)
      _escribirCeldaFuego(_calorDer[c], r);
  }
}

// Cuenta líneas donde 'ficha' aparece 2 veces y queda 1 hueco libre.
// Robot=2, Humano=1  (de app_endgame: ganador==2 → "ROBOT WINS!")
/** @brief Cuenta las líneas del tablero donde la ficha dada aparece dos veces con una celda vacía (amenazas de horquilla).
 *  @param ficha Valor de la ficha a analizar (1 = humano, 2 = robot).
 *  @return Número de líneas amenazantes (0–8). */
static int8_t _amenazas(int ficha)
{
  static const uint8_t L[8][3][2] = {
    {{0,0},{0,1},{0,2}}, {{1,0},{1,1},{1,2}}, {{2,0},{2,1},{2,2}},
    {{0,0},{1,0},{2,0}}, {{0,1},{1,1},{2,1}}, {{0,2},{1,2},{2,2}},
    {{0,0},{1,1},{2,2}}, {{0,2},{1,1},{2,0}}
  };
  int8_t n = 0;
  for (int l = 0; l < 8; l++)
  {
    int cnt = 0, vac = 0;
    for (int k = 0; k < 3; k++)
    {
      int v = tablero[L[l][k][0]][L[l][k][1]];
      if      (v == ficha) cnt++;
      else if (v == 0)     vac++;
    }
    if (cnt == 2 && vac == 1) n++;
  }
  return n;
}

/** @brief Dibuja la barra lateral de la cara del robot en el LCD durante el modo automático.
 *  Muestra ojos animados con expresiones (parpadeo, reacciones a eventos del tablero), un indicador
 *  de turno y bordes de fuego cuando quedan 3 celdas o menos. Solo se ejecuta en modoAutomatico. */
void dibujarDecoracionTurnoAuto()
{
  if (!modoAutomatico) return;

  static bool          _fuegoActivo     = false;
  static bool          _prevTurno       = true;
  // Reacciones: 0=base 1=robot_jugó 2=humano_jugó 3=robot_amenaza 4=humano_amenaza
  static uint8_t       _reaccion        = 0;
  static unsigned long _tReaccion       = 0;
  static bool          _prevRobAmenaza  = false;
  static bool          _prevHumAmenaza  = false;

  // ── 1. Fuego si quedan ≤ 3 casillas libres ──────────────────
  int libres = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (tablero[i][j] == 0) libres++;

  if (libres <= 3)
  {
    if (!_fuegoActivo)
    {
      _cargarCharsFuego();
      _fuegoActivo = true;
      // buzzerPlay se llama desde el loop principal para no bloquear lcdMutex
    }
    dibujarFuego();
    return;
  }
  if (_fuegoActivo)
  {
    lcd.createChar(5, charEngranaje1);
    lcd.createChar(6, charEngranaje2);
    lcd.createChar(7, charSnakeHead);
    _fuegoActivo = false;
  }

  // ── 2. Reacciones por cambio de turno (prioridad alta) ──────
  if (_prevTurno != turnoMaquina)
  {
    _reaccion  = turnoMaquina ? 2 : 1;
    _tReaccion = millis();
    _prevTurno = turnoMaquina;
  }

  // ── 3. Análisis del tablero (one-shot: solo al detectar cambio)
  bool robAmenaza = (_amenazas(2) > 0);
  bool humAmenaza = (_amenazas(1) > 0);

  if (_reaccion == 0 || _reaccion == 3 || _reaccion == 4)
  {
    if (robAmenaza && !_prevRobAmenaza && !humAmenaza)
    {
      _reaccion  = 3;
      _tReaccion = millis();
    }
    else if (humAmenaza && !_prevHumAmenaza)
    {
      _reaccion  = 4;
      _tReaccion = millis();
    }
  }
  _prevRobAmenaza = robAmenaza;
  _prevHumAmenaza = humAmenaza;

  if (_reaccion != 0 && millis() - _tReaccion > 1500)
    _reaccion = 0;

  // ── 4. Parpadeo siempre activo ───────────────────────────────
  // Intervalo irregular: normalmente largo, con ráfagas cortas ocasionales.
  // Cuando el humano amenaza, intervalos más cortos (nerviosismo).
  unsigned long ahora = millis();
  unsigned long iMin  = (_reaccion == 4) ? 500  : 1200;
  unsigned long iMax  = (_reaccion == 4) ? 1800 : 5000;

  if (proximoParpadeoRobot == 0)
    proximoParpadeoRobot = ahora + (unsigned long)random(iMin, iMax);

  if (!robotParpadeando && ahora >= proximoParpadeoRobot)
  {
    robotParpadeando    = true;
    finParpadeoRobot    = ahora + (unsigned long)random(80, 160);
    robotDoblePendiente = (random(100) < 40);   // 40 % → doble parpadeo rápido
    if (robotDoblePendiente)
      segundoParpadeoRobot = finParpadeoRobot + (unsigned long)random(70, 180);
  }
  if (robotParpadeando && ahora >= finParpadeoRobot)
  {
    robotParpadeando = false;
    if (!robotDoblePendiente)
    {
      // 20 % de ráfaga: próximo parpadeo muy pronto (dos parpadeos sueltos seguidos)
      if (random(100) < 20)
        proximoParpadeoRobot = ahora + (unsigned long)random(180, 450);
      else
        proximoParpadeoRobot = ahora + (unsigned long)random(iMin, iMax);
    }
  }
  if (!robotParpadeando && robotDoblePendiente && ahora >= segundoParpadeoRobot)
  {
    robotParpadeando     = true;
    robotDoblePendiente  = false;
    finParpadeoRobot     = ahora + (unsigned long)random(70, 140);
    proximoParpadeoRobot = finParpadeoRobot + (unsigned long)random(iMin, iMax);
  }

  // ── 5. Elegir expresión ──────────────────────────────────────
  // Prioridad: parpadeo > reacción (timed 1.5s) > base turno
  //  >_<  parpadeo (gana siempre, ~100 ms)
  //  ^_^  robot acaba de jugar → contento
  //  O_o  humano acaba de jugar → sorprendido
  //  *_*  robot detectó su propio 2-en-raya → emocionado (one-shot 1.5s)
  //  o_O  robot detectó amenaza del humano → alarmado  (one-shot 1.5s)
  //  O_O  turno robot, sin reacción → calculando
  //  ._.  turno humano, sin reacción → esperando
  const char* ojos;
  if      (robotParpadeando) ojos = ">_<";
  else if (_reaccion == 1)   ojos = "^_^";
  else if (_reaccion == 2)   ojos = "O_o";
  else if (_reaccion == 3)   ojos = "*_*";
  else if (_reaccion == 4)   ojos = "o_O";
  else if (turnoMaquina)     ojos = "O_O";
  else                       ojos = "._.";

  // ── 6. Dibujar robot (cols 1-3, filas 1-3) ──────────────────
  lcd.setCursor(1, 1); lcd.print("_|_");
  lcd.setCursor(1, 2); lcd.print(ojos);
  lcd.setCursor(1, 3); lcd.print("|:|");

  // ── 7. Indicador H/R (cols 13-19 = 7 chars, filas 1-3) ──────
  bool pulso = (millis() / 500) % 2 == 0;
  char letra  = turnoMaquina ? 'R' : 'H';
  char lbl[6];
  if (pulso) { lbl[0]='['; lbl[1]=' '; lbl[2]=letra; lbl[3]=' '; lbl[4]=']'; lbl[5]='\0'; }
  else       { lbl[0]=' '; lbl[1]=' '; lbl[2]=letra; lbl[3]=' '; lbl[4]=' '; lbl[5]='\0'; }

  lcd.setCursor(13, 1); lcd.print(pulso ? "  .--. " : "       ");
  lcd.setCursor(13, 2); lcd.print(" "); lcd.print(lbl); lcd.print(" ");
  lcd.setCursor(13, 3); lcd.print(pulso ? "  '--' " : "       ");
}


