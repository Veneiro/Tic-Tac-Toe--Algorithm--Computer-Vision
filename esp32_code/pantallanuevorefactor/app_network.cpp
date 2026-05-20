#include "app_contracts.h"
#include <WiFi.h>
#include <HTTPClient.h>

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

