#include "app_contracts.h"
#include <WiFi.h>
#include <HTTPClient.h>

namespace
{
  bool extraerBoolJson(const String &json, const String &clave, bool &valor)
  {
    String patron = "\"" + clave + "\"";
    int inicioClave = json.indexOf(patron);
    if (inicioClave < 0) return false;
    int inicioValor = json.indexOf(':', inicioClave + patron.length());
    if (inicioValor < 0) return false;
    int s = inicioValor + 1;
    while (s < (int)json.length() && json[s] == ' ') s++;
    if (json.substring(s, s + 4) == "true")  { valor = true;  return true; }
    if (json.substring(s, s + 5) == "false") { valor = false; return true; }
    return false;
  }

  bool extraerCadenaJson(const String &json, const String &clave, String &valor)
  {
    String patron = "\"" + clave + "\"";
    int inicioClave = json.indexOf(patron);
    if (inicioClave < 0)
    {
      return false;
    }

    int inicioValor = json.indexOf('"', inicioClave + patron.length());
    if (inicioValor < 0)
    {
      return false;
    }

    int finValor = json.indexOf('"', inicioValor + 1);
    if (finValor < 0)
    {
      return false;
    }

    valor = json.substring(inicioValor + 1, finValor);
    return true;
  }

  bool extraerEnteroJson(const String &json, const String &clave, int &valor)
  {
    String patron = "\"" + clave + "\"";
    int inicioClave = json.indexOf(patron);
    if (inicioClave < 0)
    {
      return false;
    }

    int inicioValor = json.indexOf(':', inicioClave + patron.length());
    if (inicioValor < 0)
    {
      return false;
    }

    int finValor = inicioValor + 1;
    while (finValor < (int)json.length() && (json[finValor] == ' ' || json[finValor] == '"'))
    {
      finValor++;
    }

    int finNumero = finValor;
    while (finNumero < (int)json.length() && (isDigit(json[finNumero]) || json[finNumero] == '-'))
    {
      finNumero++;
    }

    if (finNumero == finValor)
    {
      return false;
    }

    valor = json.substring(finValor, finNumero).toInt();
    return true;
  }

  bool extraerMovimientoJson(const String &json, int &fila, int &columna)
  {
    String patron = "\"movimiento\"";
    int inicioMovimiento = json.indexOf(patron);
    if (inicioMovimiento < 0)
    {
      return false;
    }

    int inicioObjeto = json.indexOf('{', inicioMovimiento + patron.length());
    if (inicioObjeto < 0)
    {
      return false;
    }

    int finObjeto = json.indexOf('}', inicioObjeto + 1);
    if (finObjeto < 0)
    {
      return false;
    }

    String movimiento = json.substring(inicioObjeto, finObjeto + 1);
    return extraerEnteroJson(movimiento, "fila", fila) && extraerEnteroJson(movimiento, "columna", columna);
  }
}

void connectToWiFi()
{
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
}

bool parsearVector(const String &src, int *arr, int n)
{
  int count = 0;
  String token = "";
  for (unsigned int i = 0; i < src.length() && count < n; i++)
  {
    char c = src[i];
    if ((c >= '0' && c <= '9') || c == '-')
    {
      token += c;
    }
    else if (token.length() > 0)
    {
      arr[count++] = token.toInt();
      token = "";
    }
  }
  if (token.length() > 0 && count < n)
    arr[count++] = token.toInt();
  return (count == n);
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
  Serial.print("Forwarding to Raspberry: ");
  Serial.println(endpoint);

  if (!http.begin(endpoint))
  {
    Serial.println("Error: Could not start HTTP connection to Raspberry");
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
      Serial.print("Raspberry response: ");
      Serial.println(response);
    }
    http.end();
    return true;
  }

  Serial.printf("HTTP error to Raspberry: %s\n", http.errorToString(httpCode).c_str());
  http.end();
  return false;
}

bool solicitarCapturaCamara()
{
  HTTPClient http;
  Serial.print("Requesting capture from ESP32-CAM: ");
  Serial.println(esp32CamURL);

  http.setTimeout(3000);
  if (!http.begin(esp32CamURL))
  {
    Serial.println("Error: could not connect to ESP32-CAM");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST("{}");

  if (httpCode <= 0)
  {
    Serial.printf("HTTP error to ESP32-CAM: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }

  String respuesta = http.getString();
  http.end();

  if (httpCode < 200 || httpCode >= 300)
  {
    Serial.printf("ESP32-CAM returned HTTP %d\n", httpCode);
    Serial.println(respuesta);
    return false;
  }

  Serial.println("Capture request sent to ESP32-CAM");
  return true;
}

void handlePedirFoto()
{
  if (solicitarCapturaCamara())
  {
    server.send(200, "text/plain", "OK - capture requested to ESP32-CAM");
    return;
  }

  server.send(500, "text/plain", "Error requesting capture to ESP32-CAM");
}

// Returns 0=OK, 1=no piece placed, 2=multiple pieces, 3=wrong color, 4=existing piece modified.
// Skips validation (returns 0) when the previous board was empty — that means the robot plays first
// and there is no human move yet to validate.
static int _validarMovHumano(int ant[3][3], int nvo[3][3])
{
  int nuevas = 0, fN = -1, cN = -1;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      if (ant[i][j] != 0 && nvo[i][j] != ant[i][j]) return 4;
      if (ant[i][j] == 0 && nvo[i][j] != 0) { nuevas++; fN = i; cN = j; }
    }
  if (nuevas == 0)
  {
    bool antVacio = true;
    for (int i = 0; i < 3 && antVacio; i++)
      for (int j = 0; j < 3 && antVacio; j++)
        if (ant[i][j] != 0) antVacio = false;
    return antVacio ? 0 : 1;
  }
  if (nuevas > 1) return 2;
  if (nvo[fN][cN] != 1) return 3;
  return 0;
}

bool procesarEntradaTablero(const String &entrada)
{
  int tableroSnapshot[3][3];
  String tableroParseado = entrada;
  int filaMovimiento = -1;
  int columnaMovimiento = -1;

  if (entrada.indexOf("\"tablero\"") >= 0)
  {
    if (!extraerCadenaJson(entrada, "tablero", tableroParseado))
    {
      Serial.println("Error: response does not contain valid 'tablero'");
      Serial.println(entrada);
      return false;
    }

    extraerMovimientoJson(entrada, filaMovimiento, columnaMovimiento);

    String leftStr  = "";
    String rightStr = "";
    if (extraerCadenaJson(entrada, "left",  leftStr))  parsearVector(leftStr,  left_fichas,  5);
    if (extraerCadenaJson(entrada, "right", rightStr)) parsearVector(rightStr, right_fichas, 5);
    extraerEnteroJson(entrada, "fuera_rojo", fuera_rojo);
    extraerEnteroJson(entrada, "fuera_azul", fuera_azul);

    bool mano = false;
    extraerBoolJson(entrada, "mano_en_tablero", mano);
    manoDetectadaEnTablero = mano;

    Serial.printf("[RX] fuera_rojo=%d  fuera_azul=%d  mano=%d\n", fuera_rojo, fuera_azul, mano);
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      tableroSnapshot[i][j] = tablero[i][j];
    }
  }

  if (!parseBoardToMatrix(tableroParseado))
  {
    Serial.println("Error: invalid board format");
    return false;
  }

  int errorVal = _validarMovHumano(tableroSnapshot, tablero);
  if (errorVal != 0)
  {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        tablero[i][j] = tableroSnapshot[i][j];

    juegoEnCurso = false;  // freeze LCD task so error message isn't overwritten
    LCD_LOCK();
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("== INVALID MOVE! ===");
    switch (errorVal)
    {
      case 1:
        lcd.setCursor(0, 1); lcd.print("  No piece added.   ");
        lcd.setCursor(0, 2); lcd.print("  Place your piece! ");
        break;
      case 2:
        lcd.setCursor(0, 1); lcd.print("  Too many pieces!  ");
        lcd.setCursor(0, 2); lcd.print("  Place only ONE.   ");
        break;
      case 3:
        lcd.setCursor(0, 1); lcd.print("  Play RED pieces   ");
        lcd.setCursor(0, 2); lcd.print("  only! (Player 1)  ");
        break;
      default:
        lcd.setCursor(0, 1); lcd.print("  Don't touch       ");
        lcd.setCursor(0, 2); lcd.print("  placed pieces!    ");
        break;
    }
    lcd.setCursor(0, 3); lcd.print("Press START to retry");
    LCD_UNLOCK();
    return false;
  }

  // Comprobamos si ya hay un ganador en el tablero recibido.
  // Si hay ganador, actualizamos la LCD y dejamos que manejarFinDeJuego se encargue.
  int ganadorDetectado = comprobarGanador();
  if (ganadorDetectado != 0)
  {
    // Parar la tarea de animación ANTES de tomar el mutex en actualizarLCD().
    // Sin esto, si la tarea está en dibujarDecoracionTurnoAuto() puede tardar
    // hasta ~3 ms en soltar el mutex; con esto ya no compite por él.
    juegoEnCurso = false;
    Serial.printf("[GAME] Winner detected in input: %d\n", ganadorDetectado);
    printBoardSerial();
    actualizarLCD();
    tableroPendiente = false;
    if (reenviarTableroRaspberry)
    {
      sendMatrixToRaspberry();
    }
    return true;
  }

  // Pintamos primero el estado reconocido por visión para que la pantalla
  // se actualice antes de que el brazo empiece a moverse.
  printBoardSerial();
  actualizarLCD();

  // Arrancar boss battle aquí: el tablero ya refleja la pieza del humano
  // (potencialmente ≤3 libres) pero el brazo aún no ha bloqueado el loop.
  // Si se esperara al check del loop principal, el brazo ya habría colocado
  // su pieza (2 libres) antes de que el check corriera.
  if (!bossBattleActivo) {
    int _libres = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (tablero[i][j] == 0) _libres++;
    if (_libres <= 3 && _libres > 0) {
      bossBattleActivo = true;
      buzzerPlay(CANCION_BOSS_BATTLE, true);
    }
  }

  if (filaMovimiento >= 0 && columnaMovimiento >= 0)
  {
    if (robotServiceMoveToCell(filaMovimiento, columnaMovimiento))
    {
      tablero[filaMovimiento][columnaMovimiento] = 2;
    }
    else
    {
      Serial.println("[ROBOT] Could not execute AI move");
    }
  }

  printBoardSerial();
  actualizarLCD();
  robotServiceApplyBoardDelta(tableroSnapshot, tablero);

  sendMatrixToRaspberry();
  return true;
}

void handleTablero()
{
  if (!server.hasArg("plain"))
  {
    server.send(400, "text/plain", "Empty body");
    Serial.println("[RX] Request without body");
    return;
  }

  if (!juegoEnCurso || !modoAutomatico)
  {
    server.send(202, "text/plain", "Ignored: not in automatic game");
    return;
  }

  if (!turnoMaquina)
  {
    server.send(202, "text/plain", "Ignored: player's turn");
    return;
  }

  tableroRecibidoHttp = server.arg("plain");
  tableroPendiente = true;

  Serial.println("\n=============================");
  Serial.println("BOARD RECEIVED FROM ESP32-CAM:");
  Serial.println(tableroRecibidoHttp);
  Serial.println("=============================\n");

  server.send(200, "text/plain", "OK - board received");
}

void handleRoot()
{
  server.send(200, "text/plain", "ESP32-S3 ready. Use POST /tablero or GET /pedir-foto");
}

bool solicitarVerificacion()
{
  String url = String(esp32CamURL);
  url.replace("/capturar", "/verificar");

  HTTPClient http;
  Serial.print("Requesting verification to ESP32-CAM: ");
  Serial.println(url);

  http.setTimeout(3000);
  if (!http.begin(url))
  {
    Serial.println("Error: could not connect to ESP32-CAM for verification");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST("{}");

  if (httpCode <= 0)
  {
    Serial.printf("HTTP error ESP32-CAM verification: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }

  http.end();
  Serial.println("Verification request sent to ESP32-CAM");
  return true;
}

void handleVerificarResultado()
{
  if (!server.hasArg("plain"))
  {
    server.send(400, "text/plain", "Empty body");
    return;
  }

  String body = server.arg("plain");
  Serial.println("[VERIFY] Result received:");
  Serial.println(body);

  bool listo  = false;
  bool limpio = false;
  bool mano   = false;

  // Parseo manual de campos bool del JSON
  auto parseBool = [&](const String& key, bool& out) {
    String pat = "\"" + key + "\"";
    int pos = body.indexOf(pat);
    if (pos < 0) return;
    int col = body.indexOf(':', pos + pat.length());
    if (col < 0) return;
    int s = col + 1;
    while (s < (int)body.length() && body[s] == ' ') s++;
    if (body.substring(s, s + 4) == "true")  out = true;
    if (body.substring(s, s + 5) == "false") out = false;
  };

  parseBool("listo",           listo);
  parseBool("tablero_limpio",  limpio);
  parseBool("mano_en_tablero", mano);

  verificarListo         = listo;
  verificarTableroLimpio = limpio;
  verificarManoEnTablero = mano;
  verificarResultadoPendiente = false;

  Serial.printf("[VERIFY] ready=%d  clean_board=%d  hand=%d\n", listo, limpio, mano);
  server.send(200, "text/plain", "OK");
}

// =========================================================
// EASTER EGG: MODO PACMAN
// =========================================================

