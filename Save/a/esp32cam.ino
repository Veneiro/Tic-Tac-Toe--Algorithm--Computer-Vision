#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "esp_http_server.h"
#include <HTTPClient.h>

// ==========================================
// CONFIGURACIÓN DE RED
// ==========================================
// const char* ssid = "Livebox6-593F";
// const char* password = "KhCSzCV5DJ4N";
// const char *ssid = "Livebox6-3935";
// const char *password = "k7R2b2TCTfxk";
//const char *ssid = "Galaxy S23 E57C";
//const char *password = "2ppyahxwjx4g7zu";
const char *ssid = "MIM-DPM-GRUPO-3";
const char *password = "mim-dpm-2026";
// Use explicit IP addresses to avoid mDNS/hostname resolution issues.
// Update the IPs below to match your network.
// Example: "http://192.168.1.28:5000/procesar"
String raspberryURL = "http://172.17.80.82:5000/procesar";
// ESP32-S3 (control board) IP and endpoint. Replace with the S3's IP if different.
String esp32s3URL = "http://192.168.100.120/tablero";

// ==========================================
// AJUSTES RÁPIDOS DE FIABILIDAD / VELOCIDAD
// Más muestras y rondas = más precisión, pero más tiempo por turno.
// ==========================================
static const int CAPTURE_SAMPLES = 5;        // 3-7 recomendado
static const int CAPTURE_ROUNDS = 3;         // 1-3 recomendado
static const int CAPTURE_MIN_VALID = 2;      // mínimo de muestras válidas para aceptar la tanda
static const int CELL_MIN_VOTES = 2;         // mínimo de votos por celda para aceptar un valor
static const int CAPTURE_LED_WARMUP_MS = 180; // tiempo para estabilizar exposición
static const int CAPTURE_GAP_MS = 80;        // pausa entre fotos dentro de una ronda

// ==========================================
// CONFIGURACIÓN DE HARDWARE ESP-EYE
// ==========================================
#define BUTTON_PIN 15  // Botón lateral ("BOOT" o "Function")
#define LED_PIN    22  // LED Blanco (Flash) suele ser 22 en ESP-EYE (prueba 21 si falla)

// Definición de pines de la cámara
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    4
#define SIOD_GPIO_NUM    18
#define SIOC_GPIO_NUM    23
#define Y9_GPIO_NUM      36
#define Y8_GPIO_NUM      37
#define Y7_GPIO_NUM      38
#define Y6_GPIO_NUM      39
#define Y5_GPIO_NUM      35
#define Y4_GPIO_NUM      14
#define Y3_GPIO_NUM      13
#define Y2_GPIO_NUM      34
#define VSYNC_GPIO_NUM   5
#define HREF_GPIO_NUM    27
#define PCLK_GPIO_NUM    25

httpd_handle_t stream_httpd = NULL;

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
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      board[r][c] = values[idx++];
    }
  }

  return true;
}

String boardToRawString(const int board[3][3]) {
  String raw = "tablero={";
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      raw += String(board[r][c]);
      if (c < 2) raw += ",";
    }
    if (r < 2) raw += ";";
  }
  raw += "}";
  return raw;
}

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

bool extractIntField(const String& json, const String& key, int& value) {
  String pattern = "\"" + key + "\"";
  int keyPos = json.indexOf(pattern);
  if (keyPos < 0) return false;

  int colonPos = json.indexOf(':', keyPos + pattern.length());
  if (colonPos < 0) return false;

  int startPos = colonPos + 1;
  while (startPos < (int)json.length() && (json[startPos] == ' ' || json[startPos] == '"')) {
    startPos++;
  }

  int endPos = startPos;
  while (endPos < (int)json.length() && (isDigit(json[endPos]) || json[endPos] == '-')) {
    endPos++;
  }

  if (endPos == startPos) return false;
  value = json.substring(startPos, endPos).toInt();
  return true;
}

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

bool capturarUnaMuestra(int board[3][3]) {
  camera_fb_t * fb = NULL;

  digitalWrite(LED_PIN, HIGH);
  delay(180);
  fb = esp_camera_fb_get();
  digitalWrite(LED_PIN, LOW);

  if (!fb) {
    return false;
  }

  HTTPClient http;
  String boardRaw = "";

  if (http.begin(raspberryURL)) {
    http.setTimeout(5000);
    http.addHeader("Content-Type", "image/jpeg");
    int httpCode = http.POST(fb->buf, fb->len);
    if (httpCode > 0) {
      boardRaw = http.getString();
    } else {
      Serial.print("[CAM] Error HTTP a Raspberry: ");
      Serial.println(http.errorToString(httpCode));
    }
    http.end();
  }

  esp_camera_fb_return(fb);

  if (boardRaw.length() == 0) {
    return false;
  }

  return parseBoardString(boardRaw, board);
}

bool capturarConsensoTablero(int boardConsenso[3][3], int &muestrasValidas) {
  int muestras[CAPTURE_SAMPLES][3][3] = {{{0}}};
  int ultimaMuestra[3][3] = {{0}};
  muestrasValidas = 0;
  int votos[3][3][3] = {{{0}}};

  for (int ronda = 0; ronda < CAPTURE_ROUNDS; ronda++) {
    for (int i = 0; i < CAPTURE_SAMPLES; i++) {
      if (!capturarUnaMuestra(muestras[i])) {
        continue;
      }

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

      if (empate) {
        mejorValor = ultimaMuestra[r][c];
      }

      boardConsenso[r][c] = mejorValor;
    }
  }

  return muestrasValidas > 0;
}

String obtenerTableroYMovimiento() {
  int board[3][3] = {{0}};
  HTTPClient http;
  String boardRaw = "";
  String movimientoJson = "";

  if (!capturarUnaMuestra(board)) {
    return "";
  }

  boardRaw = boardToRawString(board);

  String payload = "{";
  payload += "\"matriz\":" + boardToMatrixJson(board);
  payload += "}";

  String movementURL = raspberryURL;
  movementURL.replace("/procesar", "/movimiento");

  if (http.begin(movementURL)) {
    http.setTimeout(2500);
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

  if (movimientoJson.length() == 0) {
    return "";
  }

  int fila = -1;
  int columna = -1;
  if (!extractIntField(movimientoJson, "fila", fila) || !extractIntField(movimientoJson, "columna", columna)) {
    return "";
  }

  String response = "{";
  response += "\"tablero_raw\":\"" + boardRaw + "\",";
  response += "\"movimiento\":{\"fila\":" + String(fila) + ",\"columna\":" + String(columna) + "}";
  response += ",\"muestras_validas\":" + String(muestrasValidas);
  response += "}";
  return response;
}

void sendBoardToESP32S3(const String& boardState) {
  if (boardState.length() == 0) {
    Serial.println("[ESP32-S3] String vacio, no se envia");
    return;
  }

  HTTPClient forwardHttp;
  Serial.print("[ESP32-S3] Enviando tablero a: ");
  Serial.println(esp32s3URL);

  if (forwardHttp.begin(esp32s3URL)) {
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
      Serial.printf("[ESP32-S3] Error al enviar: %s\n", forwardHttp.errorToString(forwardCode).c_str());
    }
    forwardHttp.end();
  } else {
    Serial.println("[ESP32-S3] No se pudo iniciar conexion HTTP");
  }
}

// Modified capture_handler to return JSON directly

esp_err_t capture_handler(httpd_req_t *req) {
  String response = obtenerTableroYMovimiento();

  if (response.length() == 0) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Error capturando o procesando tablero");
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, response.c_str(), response.length());
}

// ==========================================
// SERVIDOR DE STREAMING (EL VISOR)
// ==========================================
esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char * part_buf[64];
  static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* _STREAM_BOUNDARY = "\r\n--frame\r\n";
  static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

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
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
      }
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      }
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  httpd_uri_t stream_uri = { .uri = "/", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capturar", .method = HTTP_POST, .handler = capture_handler, .user_ctx = NULL };
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &capture_uri);
  }
}

// ==========================================
// FUNCIÓN DE CAPTURA Y ENVÍO
// ==========================================
void captureAndSend() {
  Serial.println(">>> Boton pulsado. Preparando captura...");
  String response = obtenerTableroYMovimiento();
  if (response.length() == 0) {
    Serial.println("Error: flujo de captura fallido");
    return;
  }

  Serial.println("\n-----------------------------");
  Serial.println("RESPUESTA RECIBIDA:");
  Serial.println(response);
  Serial.println("-----------------------------\n");

  sendBoardToESP32S3(response);

  delay(1000); // Evitar rebotes
}

void setup() {
  Serial.begin(115200);
  
  // Configurar Pines (Botón y LED)
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 

  // Configurar Cámara
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  
  config.pixel_format = PIXFORMAT_JPEG;
  
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA; // 800x600
    config.jpeg_quality = 12; 
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Inicializar cámara
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error al iniciar cámara 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);
  }

  // Conectar WiFi
  WiFi.setHostname("esp32cam");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi Conectado");
  Serial.print("IP local ESP32-CAM: ");
  Serial.println(WiFi.localIP());
  Serial.print("Gateway ESP32-CAM: ");
  Serial.println(WiFi.gatewayIP());
  Serial.print("Mascara ESP32-CAM: ");
  Serial.println(WiFi.subnetMask());

  // Avoid relying on mDNS/hostname. Use fixed IPs instead.
  // If you need mDNS for debugging, uncomment the lines below.
  // if (MDNS.begin("esp32cam")) {
  //   Serial.println("mDNS iniciado: esp32cam.local");
  // }

  // Iniciar el visor web para que puedas apuntar
  Serial.print("Visor listo en: http://");
  Serial.println(WiFi.localIP());
  Serial.print("ESP32-CAM IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Endpoint captura: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capturar");
  
  startCameraServer();
  Serial.println("Servidor HTTP de la ESP32-CAM iniciado");
}

void loop() {
  // Leer botón (LOW significa pulsado)
  if (digitalRead(BUTTON_PIN) == LOW) {
    // Si se pulsa, detenemos el stream momentáneamente para procesar
    captureAndSend();
  }
  delay(50);
}