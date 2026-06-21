#include "app_contracts.h"

void esperarLiberacionBoton(int pin)
{
	while (digitalRead(pin) == HIGH)
	{
		server.handleClient();
		delay(10);
	}
	delay(50);
}

void setSemaforo(bool rojo, bool amarillo, bool verde)
{
	digitalWrite(pinLedRojo, rojo ? HIGH : LOW);
	digitalWrite(pinLedAmarillo, amarillo ? HIGH : LOW);
	digitalWrite(pinLedVerde, verde ? HIGH : LOW);
}

void cargarCaracteresBase()
{
	lcd.createChar(0, charRobot);
	lcd.createChar(1, charTrofeo);
	lcd.createChar(2, charJoy);
	lcd.createChar(3, charAparicion);
	lcd.createChar(4, charBarraInvertida);
	lcd.createChar(5, charEngranaje1);
	lcd.createChar(6, charEngranaje2);
	lcd.createChar(7, charSnakeHead);
}

void dibujarDecoracionTurnoAuto()
{
	if (!modoAutomatico)
		return;

	if (turnoMaquina)
	{
		unsigned long ahora = millis();

		if (proximoParpadeoRobot == 0)
		{
			proximoParpadeoRobot = ahora + (unsigned long)random(1400, 4200);
		}

		if (!robotParpadeando && ahora >= proximoParpadeoRobot)
		{
			robotParpadeando = true;
			finParpadeoRobot = ahora + (unsigned long)random(90, 170);
			if (random(100) < 35)
			{
				robotDoblePendiente = true;
				segundoParpadeoRobot = finParpadeoRobot + (unsigned long)random(90, 220);
			}
			else
			{
				robotDoblePendiente = false;
			}
		}

		if (robotParpadeando && ahora >= finParpadeoRobot)
		{
			robotParpadeando = false;
			if (!robotDoblePendiente)
			{
				proximoParpadeoRobot = ahora + (unsigned long)random(1400, 4800);
			}
		}

		if (!robotParpadeando && robotDoblePendiente && ahora >= segundoParpadeoRobot)
		{
			robotParpadeando = true;
			robotDoblePendiente = false;
			finParpadeoRobot = ahora + (unsigned long)random(80, 150);
			proximoParpadeoRobot = finParpadeoRobot + (unsigned long)random(1600, 5200);
		}

		bool ojosCerrados = robotParpadeando;

		lcd.setCursor(1, 1); lcd.print("_|_");
		lcd.setCursor(1, 2); lcd.print(ojosCerrados ? ">_<" : "O_O");
		lcd.setCursor(1, 3); lcd.print("|:|");

		lcd.setCursor(14, 1); lcd.print("      ");
		lcd.setCursor(14, 2); lcd.print("      ");
		lcd.setCursor(14, 3); lcd.print("      ");
	}
	else
	{
		robotParpadeando = false;
		robotDoblePendiente = false;
		proximoParpadeoRobot = millis() + (unsigned long)random(1200, 3600);

		lcd.setCursor(0, 1); lcd.print("     ");
		lcd.setCursor(0, 2); lcd.print("     ");
		lcd.setCursor(0, 3); lcd.print("     ");

		lcd.setCursor(15, 1); lcd.print(" o ");
		lcd.setCursor(15, 2); lcd.print("/|"); lcd.write(4);
		lcd.setCursor(15, 3); lcd.print("/ "); lcd.write(4);
	}
}
