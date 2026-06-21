#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>

const char* ssid = "Livebox6-593F";
const char* password = "KhCSzCV5DJ4N";
const char* raspberryPi_IP = "192.168.1.14";  // Cambia esta IP
const int raspberryPi_PORT = 5000;  // Puerto en Raspberry Pi

int matrix[3][3];

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    connectToWiFi();
    Serial.println("ESP32-S3 iniciado. Ingresa matriz 3x3 (9 números):");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        
        if (parseMatrix(input)) {
            printMatrix();
            sendMatrixToRaspberry();
        } else {
            Serial.println("Error: Ingresa 9 números separados por espacios");
        }
    }
}

void connectToWiFi() {
    Serial.print("Conectando a WiFi: ");
    Serial.println(ssid);
    
    WiFi.begin(ssid, password);
    int attempts = 0;
    
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConectado!");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nError de conexión WiFi");
    }
}

bool parseMatrix(String input) {
    int values[9];
    int count = 0;
    int lastIndex = 0;
    
    for (int i = 0; i <= input.length(); i++) {
        if (input[i] == ' ' || i == input.length()) {
            if (i > lastIndex) {
                values[count] = input.substring(lastIndex, i).toInt();
                count++;
            }
            lastIndex = i + 1;
        }
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
    Serial.println("Matriz recibida:");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Serial.print(matrix[i][j]);
            Serial.print(" ");
        }
        Serial.println();
    }
}

void sendMatrixToRaspberry() {
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

    // Se empaqueta la matriz en JSON justo antes de enviarla.
    String payload = "{\"matriz\":" + matrizJson + "}";

    String endpoint = "http://" + String(raspberryPi_IP) + ":" + String(raspberryPi_PORT) + "/movimiento";
    HTTPClient http;

    Serial.print("Enviando matriz a: ");
    Serial.println(endpoint);

    if (!http.begin(endpoint)) {
        Serial.println("Error: No se pudo iniciar conexion HTTP");
        return;
    }

    http.addHeader("Content-Type", "application/json");
    int httpCode = http.POST((uint8_t*)payload.c_str(), payload.length());

    if (httpCode > 0) {
        Serial.printf("HTTP %d\n", httpCode);
        String response = http.getString();
        if (response.length() > 0) {
            Serial.print("Respuesta servidor: ");
            Serial.println(response);
        }
        Serial.println("Matriz enviada exitosamente");
    } else {
        Serial.printf("Error HTTP: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
}