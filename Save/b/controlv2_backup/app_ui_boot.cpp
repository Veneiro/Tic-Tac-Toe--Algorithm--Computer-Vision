#include "app_contracts.h"
#include <WiFi.h>

void mostrarBienvenida()
{
	lcd.clear();

	float starX[8] = {10, 10, 10, 10, 9, 9, 9, 9};
	int starY[8]   = {0, 1, 2, 3, 0, 1, 2, 3};
	float speed[8] = {0.4, 0.8, 0.5, 0.9, -0.6, -0.4, -1.0, -0.5};

	String linea1 = "TIC-TAC-TOE";     
	String linea2 = "CONNECTING...";   

	unsigned long startTime = millis();
	unsigned long duracionMinimaAnimacion = 8000;
	unsigned long ultimoPuntoSerial = 0;

	while ((millis() - startTime < duracionMinimaAnimacion) || (WiFi.status() != WL_CONNECTED))
	{
		unsigned long tiempoTranscurrido = millis() - startTime;
		bool wifiConectada = (WiFi.status() == WL_CONNECTED);
		int letrasLinea1 = 0;
		int letrasLinea2 = 0;

		if (tiempoTranscurrido > 1500) {
			letrasLinea1 = (tiempoTranscurrido - 1500) / 300; 
			if (letrasLinea1 > 11) letrasLinea1 = 11; 
		}
    
		if (tiempoTranscurrido > 5000) {
			letrasLinea2 = (tiempoTranscurrido - 5000) / 200; 
			if (letrasLinea2 > 13) letrasLinea2 = 13;
		}

		if (tiempoTranscurrido > duracionMinimaAnimacion && !wifiConectada) {
			letrasLinea1 = 11;
			letrasLinea2 = 13;
		}

		int cursorX = -1;
		int cursorY = -1;
		bool showCursor = (tiempoTranscurrido / 250) % 2 == 0;

		if (tiempoTranscurrido > 1500 && tiempoTranscurrido <= 5000) {
			cursorX = 4 + letrasLinea1;
			cursorY = 1;
		} else if (tiempoTranscurrido > 5000) {
			cursorX = 3 + letrasLinea2;
			cursorY = 2;
		}
		if (cursorX >= 20) cursorX = 19;

		for (int i = 0; i < 8; i++) 
		{
			int oldX = (int)starX[i];
			bool chocaTexto = false;
			if (starY[i] == 1 && oldX >= 4 && oldX < 4 + letrasLinea1) chocaTexto = true;
			if (starY[i] == 2 && oldX >= 3 && oldX < 3 + letrasLinea2) chocaTexto = true;
			if (showCursor && starY[i] == cursorY && oldX == cursorX) chocaTexto = true;
                         
			if (oldX >= 0 && oldX < 20 && !chocaTexto) {
				lcd.setCursor(oldX, starY[i]); lcd.print(" ");
			}

			starX[i] += speed[i];
			if (starX[i] < 0 || starX[i] >= 20) {
				starX[i] = (speed[i] > 0) ? 10 : 9; 
			}

			int newX = (int)starX[i];
			chocaTexto = false;
			if (starY[i] == 1 && newX >= 4 && newX < 4 + letrasLinea1) chocaTexto = true;
			if (starY[i] == 2 && newX >= 3 && newX < 3 + letrasLinea2) chocaTexto = true;
			if (showCursor && starY[i] == cursorY && newX == cursorX) chocaTexto = true;

			if (newX >= 0 && newX < 20 && !chocaTexto) {
				lcd.setCursor(newX, starY[i]);
				int dist = abs(newX - 9);
				if (dist > 6) lcd.print("*");
				else if (dist > 3) lcd.print("-");
				else lcd.print(".");
			}
		}

		if (letrasLinea1 > 0) {
			lcd.setCursor(4, 1); lcd.print(linea1.substring(0, letrasLinea1));
		}
		if (letrasLinea2 > 0) {
			lcd.setCursor(3, 2); lcd.print(linea2.substring(0, letrasLinea2));
		}

		if (cursorX != -1 && cursorY != -1) {
			lcd.setCursor(cursorX, cursorY);
			if (showCursor) {
				lcd.write(255);
			} else {
				lcd.print(" ");
			}
		}

		if (!wifiConectada && millis() - ultimoPuntoSerial >= 500) {
			Serial.print(".");
			ultimoPuntoSerial = millis();
		}

		delay(50); 
	}

	Serial.println("\nWiFi conectado");
	Serial.print("IP del ESP32-S3: ");
	Serial.println(WiFi.localIP());

	transicionBarrido();
	lcd.clear();
}

void transicionBarrido() {
	for (int col = 0; col < 20; col++) {
		for (int row = 0; row < 4; row++) {
			lcd.setCursor(col, row);
			lcd.write(255); 
		}
		delay(15);
	}
	for (int col = 0; col < 20; col++) {
		for (int row = 0; row < 4; row++) {
			lcd.setCursor(col, row);
			lcd.print(" ");
		}
		delay(10);
	}
}
