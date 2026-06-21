#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

// ==========================================
// CONFIGURACION DE RED
// ==========================================
const char* ssid = "Livebox6-593F";
const char* password = "KhCSzCV5DJ4N";

const char* raspberryPi_IP = "192.168.1.14";
const int raspberryPi_PORT = 5000;

WebServer server(80);
int matrix[3][3];

void connectToWiFi() {
  Serial.print("Conectando a WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado");
  Serial.print("IP del ESP32-S3: ");
  Serial.println(WiFi.localIP());
}

bool parseBoardToMatrix(const String& input) {
  int values[9];
  int count = 0;
  String token = "";

  for (unsigned int i = 0; i < input.length(); i++) {
    char c = input[i];
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
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix[i][j] = values[idx++];
    }
  }

  return true;
}

void printMatrix() {
  Serial.println("Matriz parseada:");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Serial.print(matrix[i][j]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

bool sendMatrixToRaspberry() {
  String matrizJson = "[";
  for (int i = 0; i < 3; i++) {
    matrizJson += "[";
    for (int j = 0; j < 3; j++) {
      matrizJson += String(matrix[i][j]);
      if (j < 2) matrizJson += ",";
    }
    matrizJson += "]";
    if (i < 2) matrizJson += ",";
  }
  matrizJson += "]";

  String payload = "{\"matriz\":" + matrizJson + "}";
  String endpoint = "http://" + String(raspberryPi_IP) + ":" + String(raspberryPi_PORT) + "/movimiento";

  HTTPClient http;
  Serial.print("Reenviando a Raspberry: ");
  Serial.println(endpoint);
  Serial.print("Payload: ");
  Serial.println(payload);

  if (!http.begin(endpoint)) {
    Serial.println("Error: No se pudo iniciar conexion HTTP con Raspberry");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST((uint8_t*)payload.c_str(), payload.length());

  if (httpCode > 0) {
    Serial.printf("Raspberry HTTP %d\n", httpCode);
    String response = http.getString();
    if (response.length() > 0) {
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

void handleTablero() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "Body vacio");
    Serial.println("[RX] Peticion sin body");
    return;
  }

  String tablero = server.arg("plain");
  Serial.println("\n=============================");
  Serial.println("TABLERO RECIBIDO DESDE ESP32-CAM:");
  Serial.println(tablero);

  if (!parseBoardToMatrix(tablero)) {
    Serial.println("Error: no se pudieron extraer 9 valores de la matriz");
    Serial.println("=============================\n");
    server.send(400, "text/plain", "Formato de tablero invalido");
    return;
  }

  printMatrix();
  bool ok = sendMatrixToRaspberry();
  Serial.println("=============================\n");

  if (ok) {
    server.send(200, "text/plain", "OK - tablero recibido y reenviado");
  } else {
    server.send(502, "text/plain", "Tablero recibido, pero fallo reenvio a Raspberry");
  }
}

void handleRoot() {
  server.send(200, "text/plain", "ESP32-S3 puente listo. Usa POST /tablero");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  connectToWiFi();

  server.on("/", HTTP_GET, handleRoot);
  server.on("/tablero", HTTP_POST, handleTablero);
  server.begin();

  Serial.println("Servidor HTTP iniciado en /tablero");
}

void loop() {
  server.handleClient();
}
