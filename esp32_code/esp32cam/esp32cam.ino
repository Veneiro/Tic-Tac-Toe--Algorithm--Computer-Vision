#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "esp_http_server.h"
#include <HTTPClient.h>

const char *ssid     = "Galaxy S23 E57C";
const char *password = "2ppyahxwjx4g7zu";

String raspberryURL = "http://10.191.81.8:5000/procesar";
String esp32s3URL   = "http://10.191.81.221/tablero";

static const int CAPTURE_SAMPLES      = 1;
static const int CAPTURE_ROUNDS       = 1;
static const int CAPTURE_MIN_VALID    = 1;
static const int CELL_MIN_VOTES       = 1;
static const int CAPTURE_LED_WARMUP_MS = 180;
static const int CAPTURE_GAP_MS       = 80;

#define BUTTON_PIN 15
#define LED_PIN    22

#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    4
#define SIOD_GPIO_NUM   18
#define SIOC_GPIO_NUM   23
#define Y9_GPIO_NUM     36
#define Y8_GPIO_NUM     37
#define Y7_GPIO_NUM     38
#define Y6_GPIO_NUM     39
#define Y5_GPIO_NUM     35
#define Y4_GPIO_NUM     14
#define Y3_GPIO_NUM     13
#define Y2_GPIO_NUM     34
#define VSYNC_GPIO_NUM   5
#define HREF_GPIO_NUM   27
#define PCLK_GPIO_NUM   25

// stream_httpd (puerto 81) y api_httpd (puerto 80) son handles separados:
// stream_handler bloquea con while(true) y no puede compartir handle con /capturar
httpd_handle_t stream_httpd = NULL;
httpd_handle_t api_httpd    = NULL;

static volatile bool captureRequested  = false;
static portMUX_TYPE  captureMux        = portMUX_INITIALIZER_UNLOCKED;
static volatile bool captureInProgress = false;
bool lastButtonState = HIGH;

static int    g_left[5]          = {0, 0, 0, 0, 0};
static int    g_right[5]         = {0, 0, 0, 0, 0};
static int    g_fuera_rojo       = 0;
static int    g_fuera_azul       = 0;
static bool   g_mano_en_tablero  = false;
static String g_tablero_str      = "";  // e.g. "{0,0,0;1,0,0;0,2,0}"

/**
 * @brief Convierte la cadena de tablero "{v,v,v;v,v,v;v,v,v}" en una matriz 3x3.
 * @param boardState Cadena con 9 valores separados por comas y punto y coma.
 * @param board      Matriz 3x3 de salida (0=vacío, 1=jugador, 2=robot).
 * @return true si se parsearon exactamente 9 valores, false en caso contrario.
 */
bool parseBoardString(const String& boardState, int board[3][3]) {
  int values[9];
  int count = 0;
  String token = "";

  for (unsigned int i = 0; i < boardState.length(); i++) {
    char c = boardState[i];
    if ((c >= '0' && c <= '9') || c == '-') {
      token += c;
    } else if (token.length() > 0) {
      if (count >= 9) return false;
      values[count++] = token.toInt();
      token = "";
    }
  }
  if (token.length() > 0) {
    if (count >= 9) return false;
    values[count++] = token.toInt();
  }
  if (count != 9) return false;

  int idx = 0;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      board[r][c] = values[idx++];

  return true;
}

/**
 * @brief Parsea una cadena "{v0,v1,...,vn-1}" en un array de n enteros.
 * @param src Cadena fuente con valores numéricos separados por comas.
 * @param arr Array de salida donde se escriben los valores.
 * @param n   Número de elementos esperados.
 * @return true si se extrajeron exactamente n valores, false en caso contrario.
 */
bool parseVectorString(const String& src, int* arr, int n) {
  int count = 0;
  String token = "";
  for (unsigned int i = 0; i < src.length() && count < n; i++) {
    char c = src[i];
    if ((c >= '0' && c <= '9') || c == '-') {
      token += c;
    } else if (token.length() > 0) {
      arr[count++] = token.toInt();
      token = "";
    }
  }
  if (token.length() > 0 && count < n)
    arr[count++] = token.toInt();
  return (count == n);
}

/**
 * @brief Serializa una matriz 3x3 en formato JSON [[a,b,c],[d,e,f],[g,h,i]].
 * @param board Matriz 3x3 de enteros a serializar.
 * @return String con la representación JSON de la matriz.
 */
String boardToMatrixJson(const int board[3][3]) {
  String json = "[";
  for (int r = 0; r < 3; r++) {
    json += "[";
    for (int c = 0; c < 3; c++) {
      json += String(board[r][c]);
      if (c < 2) json += ",";
    }
    json += "]";
    if (r < 2) json += ",";
  }
  json += "]";
  return json;
}

/**
 * @brief Extrae un campo entero de una cadena JSON plana.
 * @param json  Cadena JSON de entrada.
 * @param key   Nombre del campo a buscar.
 * @param value Referencia donde se escribe el valor encontrado.
 * @return true si el campo se encontró y se pudo parsear, false en caso contrario.
 */
bool extractIntField(const String& json, const String& key, int& value) {
  String pattern = "\"" + key + "\"";
  int keyPos = json.indexOf(pattern);
  if (keyPos < 0) return false;

  int colonPos = json.indexOf(':', keyPos + pattern.length());
  if (colonPos < 0) return false;

  int startPos = colonPos + 1;
  while (startPos < (int)json.length() &&
         (json[startPos] == ' ' || json[startPos] == '"')) startPos++;

  int endPos = startPos;
  while (endPos < (int)json.length() &&
         (isDigit(json[endPos]) || json[endPos] == '-')) endPos++;

  if (endPos == startPos) return false;
  value = json.substring(startPos, endPos).toInt();
  return true;
}

/**
 * @brief Extrae el valor de un campo de cadena de una cadena JSON plana.
 * @param json Cadena JSON de entrada.
 * @param key  Nombre del campo a buscar.
 * @return Valor del campo como String, o cadena vacía si no se encuentra.
 */
String extractStringField(const String& json, const String& key) {
  String pattern = "\"" + key + "\"";
  int keyPos = json.indexOf(pattern);
  if (keyPos < 0) return "";

  int colonPos = json.indexOf(':', keyPos + pattern.length());
  if (colonPos < 0) return "";

  int startPos = json.indexOf('"', colonPos + 1);
  if (startPos < 0) return "";

  int endPos = json.indexOf('"', startPos + 1);
  if (endPos < 0) return "";

  return json.substring(startPos + 1, endPos);
}

/**
 * @brief Extrae un campo booleano (true/false) de una cadena JSON plana.
 * @param json  Cadena JSON de entrada.
 * @param key   Nombre del campo a buscar.
 * @param value Referencia donde se escribe el booleano encontrado.
 * @return true si el campo se encontró y contiene "true" o "false", false en caso contrario.
 */
bool extractBoolField(const String& json, const String& key, bool& value) {
  String pattern = "\"" + key + "\"";
  int keyPos = json.indexOf(pattern);
  if (keyPos < 0) return false;
  int colonPos = json.indexOf(':', keyPos + pattern.length());
  if (colonPos < 0) return false;
  int startPos = colonPos + 1;
  while (startPos < (int)json.length() && json[startPos] == ' ') startPos++;
  if (json.substring(startPos, startPos + 4) == "true")  { value = true;  return true; }
  if (json.substring(startPos, startPos + 5) == "false") { value = false; return true; }
  return false;
}

/**
 * @brief Captura un fotograma estable descartando el primer frame del sensor.
 * @return Puntero al frame buffer capturado, o NULL si falla la cámara.
 */
camera_fb_t *capturarFotoEstable() {
  digitalWrite(LED_PIN, HIGH);
  delay(CAPTURE_LED_WARMUP_MS);

  // Descartar primer frame (ajuste automático del sensor)
  camera_fb_t *frameDescartado = esp_camera_fb_get();
  if (frameDescartado != NULL) esp_camera_fb_return(frameDescartado);

  delay(CAPTURE_GAP_MS);
  camera_fb_t *fb = esp_camera_fb_get();
  digitalWrite(LED_PIN, LOW);
  return fb;
}

/**
 * @brief Captura una foto y la envía a la Raspberry Pi para obtener el estado del tablero.
 * @param board Matriz 3x3 de salida con el estado detectado del tablero.
 * @return true si la captura y el análisis se completaron correctamente, false en caso de error.
 */
bool capturarUnaMuestra(int board[3][3]) {
  camera_fb_t *fb = capturarFotoEstable();
  if (!fb) {
    Serial.println("[CAM] ERROR: esp_camera_fb_get() devolvio NULL - camara ocupada o no inicializada");
    return false;
  }

  Serial.printf("[CAM] Foto OK (%u bytes) → enviando a %s\n", fb->len, raspberryURL.c_str());

  HTTPClient http;
  String respuesta = "";

  if (!http.begin(raspberryURL)) {
    Serial.println("[CAM] ERROR: http.begin() fallo - URL invalida?");
    esp_camera_fb_return(fb);
    return false;
  }

  http.setTimeout(6000);
  http.addHeader("Content-Type", "image/jpeg");
  int httpCode = http.POST(fb->buf, fb->len);

  if (httpCode > 0) {
    Serial.printf("[CAM] Raspberry /procesar respondio HTTP %d\n", httpCode);
    respuesta = http.getString();
  } else {
    Serial.printf("[CAM] ERROR HTTP a Raspberry /procesar: %s (codigo %d)\n",
                  http.errorToString(httpCode).c_str(), httpCode);
  }
  http.end();

  esp_camera_fb_return(fb);

  if (respuesta.length() == 0) {
    Serial.println("[CAM] ERROR: respuesta vacia de Raspberry");
    return false;
  }

  // Formato nuevo: {"tablero":"{...}", "left":"{...}", "right":"{...}", ...}
  String tableroStr = extractStringField(respuesta, "tablero");
  if (tableroStr.length() == 0) {
    Serial.printf("[CAM] ERROR: respuesta sin campo 'tablero' (formato incorrecto?): %.120s\n", respuesta.c_str());
    return false;
  }
  if (!parseBoardString(tableroStr, board)) {
    Serial.printf("[CAM] ERROR: no se pudo parsear tablero: %s\n", tableroStr.c_str());
    return false;
  }

  g_tablero_str = tableroStr;

  String leftStr  = extractStringField(respuesta, "left");
  String rightStr = extractStringField(respuesta, "right");
  if (leftStr.length()  > 0) parseVectorString(leftStr,  g_left,  5);
  if (rightStr.length() > 0) parseVectorString(rightStr, g_right, 5);
  extractIntField(respuesta,  "fuera_rojo",      g_fuera_rojo);
  extractIntField(respuesta,  "fuera_azul",      g_fuera_azul);
  extractBoolField(respuesta, "mano_en_tablero", g_mano_en_tablero);

  return true;
}

/**
 * @brief Toma varias muestras del tablero y determina el estado por votación por celda.
 * @param boardConsenso   Matriz 3x3 de salida con el valor más votado por celda.
 * @param muestrasValidas Referencia donde se escribe el número de muestras correctas obtenidas.
 * @return true si se obtuvo al menos una muestra válida, false si todas fallaron.
 */
bool capturarConsensoTablero(int boardConsenso[3][3], int &muestrasValidas) {
  int muestras[CAPTURE_SAMPLES][3][3] = {{{0}}};
  int ultimaMuestra[3][3] = {{0}};
  muestrasValidas = 0;
  int votos[3][3][3] = {{{0}}};

  for (int ronda = 0; ronda < CAPTURE_ROUNDS; ronda++) {
    for (int i = 0; i < CAPTURE_SAMPLES; i++) {
      if (!capturarUnaMuestra(muestras[i])) continue;

      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          int valor = muestras[i][r][c];
          if (valor >= 0 && valor <= 2) {
            votos[r][c][valor]++;
            ultimaMuestra[r][c] = valor;
          }
        }
      }
      muestrasValidas++;
    }
  }

  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      int mejorValor = 0;
      int mejorVotos = votos[r][c][0];
      bool empate = false;

      for (int valor = 1; valor <= 2; valor++) {
        if (votos[r][c][valor] > mejorVotos) {
          mejorVotos = votos[r][c][valor];
          mejorValor = valor;
          empate = false;
        } else if (votos[r][c][valor] == mejorVotos && votos[r][c][valor] > 0) {
          empate = true;
        }
      }
      if (empate) mejorValor = ultimaMuestra[r][c];
      boardConsenso[r][c] = mejorValor;
    }
  }

  return muestrasValidas > 0;
}

/**
 * @brief Ejecuta el flujo completo: captura consenso del tablero, consulta a la IA
 *        el siguiente movimiento y construye el JSON de respuesta.
 * @return JSON string con tablero, movimiento, laterales y flags, o cadena vacía si falla.
 */
String obtenerTableroYMovimiento() {
  int board[3][3] = {{0}};
  int muestrasValidas = 0;
  HTTPClient http;
  String movimientoJson = "";

  if (!capturarConsensoTablero(board, muestrasValidas)) return "";

  String payload = "{\"matriz\":" + boardToMatrixJson(board) + "}";

  String movementURL = raspberryURL;
  movementURL.replace("/procesar", "/movimiento");

  if (http.begin(movementURL)) {
    http.setTimeout(6000);  // FIX: 6 s
    http.addHeader("Content-Type", "application/json");
    int httpCode = http.POST((uint8_t*)payload.c_str(), payload.length());
    if (httpCode > 0) {
      movimientoJson = http.getString();
    } else {
      Serial.print("[CAM] Error HTTP a Raspberry: ");
      Serial.println(http.errorToString(httpCode));
    }
    http.end();
  }

  if (movimientoJson.length() == 0) return "";

  int fila = -1, columna = -1;
  bool hayMovimiento = extractIntField(movimientoJson, "fila", fila) &&
                       extractIntField(movimientoJson, "columna", columna);
  if (!hayMovimiento) {
    // Tablero lleno: la IA dice "Fin" sin movimiento (empate o ganador ya visible).
    // Enviamos igualmente el tablero para que el ESP32-S3 detecte el fin de partida.
    Serial.println("[CAM] Raspberry dice Fin (sin movimiento) - enviando tablero para deteccion");
  }

  // Vector left
  String leftJson = "{";
  for (int i = 0; i < 5; i++) { leftJson += String(g_left[i]); if (i < 4) leftJson += ","; }
  leftJson += "}";

  // Vector right
  String rightJson = "{";
  for (int i = 0; i < 5; i++) { rightJson += String(g_right[i]); if (i < 4) rightJson += ","; }
  rightJson += "}";

  String response = "{";
  response += "\"tablero\":\"" + g_tablero_str + "\",";
  response += "\"left\":\"" + leftJson + "\",";
  response += "\"right\":\"" + rightJson + "\",";
  response += "\"fuera_rojo\":" + String(g_fuera_rojo) + ",";
  response += "\"fuera_azul\":" + String(g_fuera_azul) + ",";
  response += "\"mano_en_tablero\":" + String(g_mano_en_tablero ? "true" : "false") + ",";
  if (hayMovimiento) {
    response += "\"movimiento\":{\"fila\":" + String(fila) + ",\"columna\":" + String(columna) + "},";
  }
  response += "\"muestras_validas\":" + String(muestrasValidas);
  response += "}";
  return response;
}

static volatile bool verificarRequested  = false;
static portMUX_TYPE  verificarMux        = portMUX_INITIALIZER_UNLOCKED;
static volatile bool verificarInProgress = false;

/**
 * @brief Captura una foto y la envía al endpoint /verificar_tablero de la Raspberry Pi.
 *        El resultado se reenvía directamente al ESP32-S3.
 */
void verificarEstadoTablero() {
  camera_fb_t *fb = capturarFotoEstable();
  if (!fb) {
    Serial.println("[VERIFICAR] ERROR: no se pudo capturar foto");
    return;
  }

  String verURL = raspberryURL;
  verURL.replace("/procesar", "/verificar_tablero");

  HTTPClient http;
  String respuesta = "";

  if (http.begin(verURL)) {
    http.setTimeout(8000);
    http.addHeader("Content-Type", "image/jpeg");
    int httpCode = http.POST(fb->buf, fb->len);
    if (httpCode > 0) {
      Serial.printf("[VERIFICAR] Raspberry /verificar_tablero → HTTP %d\n", httpCode);
      respuesta = http.getString();
    } else {
      Serial.printf("[VERIFICAR] Error HTTP: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }
  esp_camera_fb_return(fb);

  if (respuesta.length() == 0) {
    Serial.println("[VERIFICAR] Sin respuesta de Raspberry");
    return;
  }

  // Reenviar resultado al ESP32-S3
  String fwdURL = esp32s3URL;
  fwdURL.replace("/tablero", "/verificar_resultado");

  HTTPClient fwdHttp;
  if (fwdHttp.begin(fwdURL)) {
    fwdHttp.setTimeout(5000);
    fwdHttp.addHeader("Content-Type", "application/json");
    int fwdCode = fwdHttp.POST((uint8_t*)respuesta.c_str(), respuesta.length());
    if (fwdCode > 0) {
      Serial.printf("[VERIFICAR] Enviado a ESP32-S3 → HTTP %d\n", fwdCode);
    } else {
      Serial.printf("[VERIFICAR] Error enviando a ESP32-S3: %s\n",
                    fwdHttp.errorToString(fwdCode).c_str());
    }
    fwdHttp.end();
  }
}

/**
 * @brief Handler HTTP para POST/GET /verificar. Programa la verificación del tablero de forma
 *        asíncrona y responde inmediatamente con 200 o 503 si ya hay una captura en curso.
 * @param req Puntero a la petición HTTP entrante.
 * @return ESP_OK si la respuesta se envió correctamente.
 */
esp_err_t verificar_handler(httpd_req_t *req) {
  if (captureInProgress || verificarInProgress) {
    httpd_resp_set_status(req, "503 Busy");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, "{\"error\":\"captura en progreso\"}", HTTPD_RESP_USE_STRLEN);
  }
  portENTER_CRITICAL(&verificarMux);
  verificarRequested = true;
  portEXIT_CRITICAL(&verificarMux);
  Serial.println("[HTTP] POST /verificar recibido: verificacion programada");
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, "{\"status\":\"verificacion programada\"}", HTTPD_RESP_USE_STRLEN);
}

/**
 * @brief Reenvía el JSON de estado del tablero y movimiento al ESP32-S3 por HTTP POST.
 * @param boardState JSON string con el estado completo a enviar.
 */
void sendBoardToESP32S3(const String& boardState) {
  if (boardState.length() == 0) {
    Serial.println("[ESP32-S3] String vacio, no se envia");
    return;
  }

  HTTPClient forwardHttp;
  Serial.print("[ESP32-S3] Enviando tablero a: ");
  Serial.println(esp32s3URL);

  if (forwardHttp.begin(esp32s3URL)) {
    forwardHttp.setTimeout(5000);
    forwardHttp.addHeader("Content-Type", "application/json");
    int forwardCode = forwardHttp.POST((uint8_t*)boardState.c_str(), boardState.length());

    if (forwardCode > 0) {
      Serial.printf("[ESP32-S3] Envio OK, HTTP %d\n", forwardCode);
      String ack = forwardHttp.getString();
      if (ack.length() > 0) {
        Serial.print("[ESP32-S3] ACK: ");
        Serial.println(ack);
      }
    } else {
      Serial.printf("[ESP32-S3] Error al enviar: %s\n",
                    forwardHttp.errorToString(forwardCode).c_str());
    }
    forwardHttp.end();
  } else {
    Serial.println("[ESP32-S3] No se pudo iniciar conexion HTTP");
  }
}

/**
 * @brief Handler HTTP para POST/GET /capturar. Programa la captura de forma asíncrona
 *        y responde inmediatamente con 200 o 503 si ya hay una captura en curso.
 * @param req Puntero a la petición HTTP entrante.
 * @return ESP_OK si la respuesta se envió correctamente.
 */
esp_err_t capture_handler(httpd_req_t *req) {
  if (captureInProgress) {
    httpd_resp_set_status(req, "503 Busy");
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, "{\"error\":\"captura en progreso\"}", HTTPD_RESP_USE_STRLEN);
  }

  portENTER_CRITICAL(&captureMux);
  captureRequested = true;
  portEXIT_CRITICAL(&captureMux);

  Serial.println("[HTTP] POST /capturar recibido: captura programada");

  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, "{\"status\":\"captura programada\"}", HTTPD_RESP_USE_STRLEN);
}

/**
 * @brief Handler MJPEG para GET /stream. Transmite frames JPEG continuos en multipart.
 *        Bloquea el hilo con un while(true) y por eso se ejecuta en un servidor separado.
 * @param req Puntero a la petición HTTP entrante.
 * @return ESP_OK al finalizar la transmisión, ESP_FAIL si la cámara falla.
 */
esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  static const char* _STREAM_CONTENT_TYPE =
      "multipart/x-mixed-replace;boundary=frame";
  static const char* _STREAM_BOUNDARY = "\r\n--frame\r\n";
  static const char* _STREAM_PART =
      "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Fallo al obtener frame para stream");
      res = ESP_FAIL;
    } else {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      if (res == ESP_OK) {
        size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
      }
      if (res == ESP_OK)
        res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
  }
  return res;
}

/**
 * @brief Inicializa los dos servidores HTTP: API REST en puerto 80 y stream MJPEG en puerto 81.
 */
void startCameraServer() {
  httpd_config_t api_config = HTTPD_DEFAULT_CONFIG();
  api_config.server_port = 80;
  api_config.stack_size  = 8192;  // stack generoso para HTTPClient anidado

  httpd_uri_t capture_post_uri = {
    .uri      = "/capturar",
    .method   = HTTP_POST,
    .handler  = capture_handler,
    .user_ctx = NULL
  };
  httpd_uri_t capture_get_uri = {
    .uri      = "/capturar",
    .method   = HTTP_GET,
    .handler  = capture_handler,
    .user_ctx = NULL
  };
  httpd_uri_t verificar_post_uri = {
    .uri      = "/verificar",
    .method   = HTTP_POST,
    .handler  = verificar_handler,
    .user_ctx = NULL
  };
  httpd_uri_t verificar_get_uri = {
    .uri      = "/verificar",
    .method   = HTTP_GET,
    .handler  = verificar_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&api_httpd, &api_config) == ESP_OK) {
    httpd_register_uri_handler(api_httpd, &capture_post_uri);
    httpd_register_uri_handler(api_httpd, &capture_get_uri);
    httpd_register_uri_handler(api_httpd, &verificar_post_uri);
    httpd_register_uri_handler(api_httpd, &verificar_get_uri);
    Serial.println("[HTTP] Servidor API iniciado en puerto 80");
  } else {
    Serial.println("[HTTP] ERROR al iniciar servidor API");
  }

  // --- Servidor Stream (puerto 81) ---
  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port      = 81;
  stream_config.ctrl_port        = 32769;  // puerto de control distinto al del API
  stream_config.stack_size       = 8192;
  stream_config.max_open_sockets = 1;      // un solo cliente de stream a la vez

  httpd_uri_t stream_uri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("[HTTP] Servidor Stream iniciado en puerto 81");
  } else {
    Serial.println("[HTTP] ERROR al iniciar servidor Stream");
  }
}

/**
 * @brief Ejecuta el ciclo completo de captura y envío iniciado desde el botón físico.
 *        Bloquea capturas simultáneas mediante la bandera captureInProgress.
 */
void captureAndSend() {
  if (captureInProgress) {
    Serial.println("[BTN] Captura ya en progreso, ignorando pulsacion");
    return;
  }

  captureInProgress = true;
  Serial.println(">>> Boton pulsado. Preparando captura...");

  String response = obtenerTableroYMovimiento();
  if (response.length() == 0) {
    Serial.println("Error: flujo de captura fallido");
    captureInProgress = false;
    return;
  }

  Serial.println("\n-----------------------------");
  Serial.println("RESPUESTA RECIBIDA:");
  Serial.println(response);
  Serial.println("-----------------------------\n");

  sendBoardToESP32S3(response);

  captureInProgress = false;
  delay(1000);  // Evitar rebotes del botón
}

void setup() {
  Serial.begin(115200);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Configurar cámara
  camera_config_t config;
  config.ledc_channel  = LEDC_CHANNEL_0;
  config.ledc_timer    = LEDC_TIMER_0;
  config.pin_d0        = Y2_GPIO_NUM;
  config.pin_d1        = Y3_GPIO_NUM;
  config.pin_d2        = Y4_GPIO_NUM;
  config.pin_d3        = Y5_GPIO_NUM;
  config.pin_d4        = Y6_GPIO_NUM;
  config.pin_d5        = Y7_GPIO_NUM;
  config.pin_d6        = Y8_GPIO_NUM;
  config.pin_d7        = Y9_GPIO_NUM;
  config.pin_xclk      = XCLK_GPIO_NUM;
  config.pin_pclk      = PCLK_GPIO_NUM;
  config.pin_vsync     = VSYNC_GPIO_NUM;
  config.pin_href      = HREF_GPIO_NUM;
  config.pin_sscb_sda  = SIOD_GPIO_NUM;
  config.pin_sscb_scl  = SIOC_GPIO_NUM;
  config.pin_pwdn      = PWDN_GPIO_NUM;
  config.pin_reset     = RESET_GPIO_NUM;
  config.xclk_freq_hz  = 20000000;
  config.pixel_format  = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size  = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count    = 2;
    config.grab_mode   = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size  = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count    = 1;
    config.grab_mode   = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error al iniciar camara: 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_hmirror(s, 0);
    s->set_vflip(s, 1);
  }

  // Conectar WiFi
  WiFi.setHostname("esp32cam");
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Conectado");
  Serial.print("IP local ESP32-CAM: ");  Serial.println(WiFi.localIP());
  Serial.print("Gateway ESP32-CAM: ");   Serial.println(WiFi.gatewayIP());
  Serial.print("Mascara ESP32-CAM: ");   Serial.println(WiFi.subnetMask());

  startCameraServer();

  Serial.println("\n=== ENDPOINTS DISPONIBLES ===");
  Serial.print("Captura (ESP32-S3 → CAM):  POST http://");
  Serial.print(WiFi.localIP()); Serial.println("/capturar");
  Serial.print("Stream (visor):            GET  http://");
  Serial.print(WiFi.localIP()); Serial.println(":81/stream");
  Serial.println("=============================\n");
}

void loop() {
  bool currentButtonState = digitalRead(BUTTON_PIN);
  bool buttonPressed = (lastButtonState == HIGH && currentButtonState == LOW);
  lastButtonState = currentButtonState;

  bool doCapture = false;
  portENTER_CRITICAL(&captureMux);
  if (captureRequested) {
    captureRequested = false;
    doCapture = true;
  }
  portEXIT_CRITICAL(&captureMux);

  bool doVerificar = false;
  portENTER_CRITICAL(&verificarMux);
  if (verificarRequested) {
    verificarRequested = false;
    doVerificar = true;
  }
  portEXIT_CRITICAL(&verificarMux);

  if ((doCapture || buttonPressed) && !captureInProgress && !verificarInProgress) {
    captureAndSend();
  } else if (doVerificar && !captureInProgress && !verificarInProgress) {
    verificarInProgress = true;
    verificarEstadoTablero();
    verificarInProgress = false;
  }

  delay(50);
}
