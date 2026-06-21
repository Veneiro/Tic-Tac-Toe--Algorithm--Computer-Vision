#include "app_contracts.h"
#include <WiFi.h>
#include <HTTPClient.h>

namespace
{
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

void handlePedirFoto()
{
	if (solicitarCapturaCamara())
	{
		server.send(200, "text/plain", "OK - captura solicitada a la ESP32-CAM");
		return;
	}

	server.send(500, "text/plain", "Error al solicitar captura a la ESP32-CAM");
}

void procesarEntradaTablero(const String &entrada)
{
	int tableroSnapshot[3][3];
	String tableroParseado = entrada;
	int filaMovimiento = -1;
	int columnaMovimiento = -1;

	if (entrada.indexOf("\"tablero_raw\"") >= 0)
	{
		if (!extraerCadenaJson(entrada, "tablero_raw", tableroParseado))
		{
			Serial.println("Error: la respuesta de la camara no contiene tablero_raw valido");
			Serial.println(entrada);
			return;
		}

		extraerMovimientoJson(entrada, filaMovimiento, columnaMovimiento);
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

	int ganadorDetectado = comprobarGanador();
	if (ganadorDetectado != 0)
	{
		Serial.printf("[GAME] Ganador detectado en la entrada: %d - deteniendo movimientos\n", ganadorDetectado);
		printBoardSerial();
		actualizarLCD();
		mostrarCopaASCII();
		juegoEnCurso = false;
		tableroPendiente = false;
		if (reenviarTableroRaspberry)
		{
			sendMatrixToRaspberry();
		}
		return;
	}

	printBoardSerial();
	actualizarLCD();

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

void handleRoot()
{
	server.send(200, "text/plain", "ESP32-S3 fusion listo. Usa POST /tablero o GET /pedir-foto");
}
