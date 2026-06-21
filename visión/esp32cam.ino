#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ── Configuración ──────────────────────────
const char* SSID     = "TU_SSID";
const char* PASSWORD = "TU_PASSWORD";

// ── Pines AI Thinker ESP32-CAM ─────────────
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
#define LED_GPIO_NUM    4

WebServer server(80);

// GET /stream → MJPEG continuo (lo que lee OpenCV con cap.read())
void handleStream() {
  WiFiClient client = server.client();

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Cache-Control: no-cache");
  client.println();

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) continue;

    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);
  }
}

// GET / → info
void handleRoot() {
  String ip = WiFi.localIP().toString();
  server.send(200, "text/plain",
    "ESP32-CAM activa\nStream: http://" + ip + "/stream");
}

void initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0       = Y2_GPIO_NUM;
  cfg.pin_d1       = Y3_GPIO_NUM;
  cfg.pin_d2       = Y4_GPIO_NUM;
  cfg.pin_d3       = Y5_GPIO_NUM;
  cfg.pin_d4       = Y6_GPIO_NUM;
  cfg.pin_d5       = Y7_GPIO_NUM;
  cfg.pin_d6       = Y8_GPIO_NUM;
  cfg.pin_d7       = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 16000000;   // 20MHz causa timeouts en algunos boards
  cfg.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_VGA;  // 640×480
    cfg.jpeg_quality = 10;
    cfg.fb_count     = 2;
  } else {
    cfg.frame_size   = FRAMESIZE_CIF;  // 400×296
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 1;
  }

  delay(100); // espera que el sensor estabilice la alimentación

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("Error cámara: 0x%x — reintentando con 10 MHz\n", err);
    cfg.xclk_freq_hz = 10000000;
    delay(200);
    err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
      Serial.printf("Error definitivo: 0x%x\n", err);
      while (true) delay(1000);
    }
  }
  Serial.println("Cámara OK");

  sensor_t* s = esp_camera_sensor_get();
  s->set_whitebal(s, 1);
  s->set_exposure_ctrl(s, 1);
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);

  initCamera();

  WiFi.begin(SSID, PASSWORD);
  Serial.print("Conectando");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\n=== LISTO ===");
  Serial.print("Stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");

  server.on("/",       HTTP_GET, handleRoot);
  server.on("/stream", HTTP_GET, handleStream);
  server.begin();

  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_GPIO_NUM, HIGH); delay(100);
    digitalWrite(LED_GPIO_NUM, LOW);  delay(100);
  }
}

void loop() {
  server.handleClient();
}
