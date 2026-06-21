#include "app_contracts.h"
#include <WiFi.h>
#include <HTTPClient.h>

namespace
{
  /** @brief Parsea un campo booleano de una cadena JSON sin usar una biblioteca externa.
   *  @param json  Cadena JSON de origen.
   *  @param clave Clave del campo a buscar.
   *  @param valor Salida: valor booleano parseado.
   *  @return Verdadero si el campo fue encontrado y parseado; falso en caso contrario. */
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

  /** @brief Parsea un campo de cadena entre comillas de una cadena JSON sin usar una biblioteca externa.
   *  @param json  Cadena JSON de origen.
   *  @param clave Clave del campo cuyo valor de cadena se extrae.
   *  @param valor Salida: valor de cadena parseado (sin las comillas delimitadoras).
   *  @return Verdadero si el campo fue encontrado y parseado; falso en caso contrario. */
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

  /** @brief Parsea un campo entero de una cadena JSON sin usar una biblioteca externa.
   *  @param json  Cadena JSON de origen.
   *  @param clave Clave del campo cuyo valor entero se extrae.
   *  @param valor Salida: valor entero parseado.
   *  @return Verdadero si el campo fue encontrado y parseado; falso en caso contrario. */
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

  /** @brief Extrae los enteros fila y columna del objeto anidado "movimiento":{} en el JSON.
   *  @param json    Cadena JSON de origen que contiene un objeto "movimiento".
   *  @param fila    Salida: índice de fila del movimiento.
   *  @param columna Salida: índice de columna del movimiento.
   *  @return Verdadero si ambos campos fueron encontrados y parseados; falso en caso contrario. */
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

/** @brief Inicia una conexión WiFi STA no bloqueante usando el ssid y la contraseña configurados. */
void connectToWiFi()
{
  Serial.print("Conectando a WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
}

/** @brief Parsea n enteros de una cadena separada por comas en arr[].
 *  @param src Cadena de origen que contiene tokens enteros separados por comas.
 *  @param arr Array de salida para almacenar los valores parseados.
 *  @param n   Número de enteros esperados.
 *  @return Verdadero si se parsearon exactamente n enteros; falso en caso contrario. */
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

/** @brief Parsea 9 enteros de una cadena de tablero y rellena la matriz global tablero[][].
 *  @param input Cadena que contiene exactamente 9 tokens de dígitos (0–2) en orden fila-mayor.
 *  @return Verdadero si se parsearon exactamente 9 valores y se rellenó la matriz; falso en caso contrario. */
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

/** @brief Envía por HTTP POST el tablero[][] actual como matriz JSON al endpoint /movimiento de la Raspberry Pi.
 *  @return Verdadero si la solicitud se envió y el servidor respondió con un código HTTP positivo; falso en caso contrario. */
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

/** @brief Envía un HTTP POST al endpoint /capturar de la ESP32-CAM para disparar una captura de foto.
 *  @return Verdadero si la cámara confirmó la solicitud (HTTP 2xx); falso en caso de error. */
bool solicitarCapturaCamara()
{
  HTTPClient http;
  Serial.print("Solicitando captura a ESP32-CAM: ");
  Serial.println(esp32CamURL);

  http.setTimeout(3000);
  if (!http.begin(esp32CamURL))
  {
    Serial.println("Error: no se pudo iniciar conexion con ESP32-CAM");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST("{}");

  if (httpCode <= 0)
  {
    Serial.printf("Error HTTP hacia ESP32-CAM: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }

  String respuesta = http.getString();
  http.end();

  if (httpCode < 200 || httpCode >= 300)
  {
    Serial.printf("ESP32-CAM devolvio HTTP %d\n", httpCode);
    Serial.println(respuesta);
    return false;
  }

  Serial.println("Disparo enviado correctamente a la ESP32-CAM");
  return true;
}

/** @brief Manejador GET /pedir-foto del WebServer: llama a solicitarCapturaCamara y devuelve el resultado como texto plano. */
void handlePedirFoto()
{
  if (solicitarCapturaCamara())
  {
    server.send(200, "text/plain", "OK - captura solicitada a la ESP32-CAM");
    return;
  }

  server.send(500, "text/plain", "Error al solicitar captura a la ESP32-CAM");
}

/** @brief Parsea el payload JSON completo recibido de la cámara, actualiza el estado del tablero,
 *  ordena al robot que se mueva y refresca el LCD.
 *  @param entrada Cadena JSON que contiene "tablero", "movimiento", arrays de piezas laterales y banderas. */
void procesarEntradaTablero(const String &entrada)
{
  int tableroSnapshot[3][3];
  String tableroParseado = entrada;
  int filaMovimiento = -1;
  int columnaMovimiento = -1;

  if (entrada.indexOf("\"tablero\"") >= 0)
  {
    if (!extraerCadenaJson(entrada, "tablero", tableroParseado))
    {
      Serial.println("Error: la respuesta no contiene 'tablero' valido");
      Serial.println(entrada);
      return;
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
    Serial.println("Error: formato de tablero invalido");
    return;
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
    Serial.printf("[GAME] Ganador detectado en la entrada: %d\n", ganadorDetectado);
    printBoardSerial();
    actualizarLCD();
    tableroPendiente = false;
    if (reenviarTableroRaspberry)
    {
      sendMatrixToRaspberry();
    }
    return;
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
      Serial.println("[ROBOT] No se pudo ejecutar el movimiento indicado por la IA");
    }
  }

  printBoardSerial();
  actualizarLCD();
  robotServiceApplyBoardDelta(tableroSnapshot, tablero);

  sendMatrixToRaspberry();
}

/** @brief Manejador POST /tablero del WebServer: valida el estado de la partida y encola el JSON del tablero recibido.
 *  Rechaza las solicitudes si la partida no está en modo automático, no está en curso o no es el turno del robot. */
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

  server.send(200, "text/plain", "OK - tablero recibido");
}

/** @brief Manejador GET / del WebServer: devuelve una cadena de estado que confirma que el ESP32-S3 está listo. */
void handleRoot()
{
  server.send(200, "text/plain", "ESP32-S3 fusion listo. Usa POST /tablero o GET /pedir-foto");
}

/** @brief Envía un HTTP POST al endpoint /verificar de la ESP32-CAM para solicitar la verificación del tablero.
 *  @return Verdadero si la solicitud se envió correctamente; falso en caso de error de conexión o HTTP. */
bool solicitarVerificacion()
{
  String url = String(esp32CamURL);
  url.replace("/capturar", "/verificar");

  HTTPClient http;
  Serial.print("Solicitando verificacion a ESP32-CAM: ");
  Serial.println(url);

  http.setTimeout(3000);
  if (!http.begin(url))
  {
    Serial.println("Error: no se pudo iniciar conexion con ESP32-CAM para verificar");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST("{}");

  if (httpCode <= 0)
  {
    Serial.printf("Error HTTP verificacion ESP32-CAM: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }

  http.end();
  Serial.println("Solicitud de verificacion enviada a ESP32-CAM");
  return true;
}

/** @brief Manejador POST /verificar_resultado del WebServer: parsea los booleanos "listo", "tablero_limpio" y
 *  "mano_en_tablero" del cuerpo JSON y actualiza las banderas globales de verificación. */
void handleVerificarResultado()
{
  if (!server.hasArg("plain"))
  {
    server.send(400, "text/plain", "Body vacio");
    return;
  }

  String body = server.arg("plain");
  Serial.println("[VERIFICAR] Resultado recibido:");
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

  Serial.printf("[VERIFICAR] listo=%d  tablero_limpio=%d  mano=%d\n", listo, limpio, mano);
  server.send(200, "text/plain", "OK");
}


