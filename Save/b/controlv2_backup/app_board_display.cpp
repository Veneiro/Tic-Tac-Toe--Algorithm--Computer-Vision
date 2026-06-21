#include "app_contracts.h"

void leetablero(String entrada)
{
	int inicio = entrada.indexOf('{');
	int fin = entrada.indexOf('}');

	if (inicio == -1 || fin == -1)
	{
		Serial.println("Error: Formato de string no valido");
		return;
	}

	String contenido = entrada.substring(inicio + 1, fin);
	int fila = 0;
	int col = 0;

	for (int i = 0; i < contenido.length(); i++)
	{
		char c = contenido.charAt(i);

		if (c == ',')
		{
			col++; 
		}
		else if (c == ';')
		{
			fila++;  
			col = 0; 
		}
		else if (c >= '0' && c <= '2')
		{
			if (fila < 3 && col < 3)
			{
				tablero[fila][col] = c - '0';
			}
		}
	}
}

void animarEntradaTablero() 
{
	lcd.clear();
	String titulo = "===  [ BOARD ]   ===";
	lcd.setCursor(0, 0);
	for(int i = 0; i < titulo.length(); i++) {
		lcd.print(titulo[i]);
		delay(30);
	}
	delay(150);

	for (int i = 1; i <= 3; i++) {
		lcd.setCursor(0, i);
		lcd.print("      [     ]       "); 
		delay(120);
	}

	for (int i = 1; i <= 3; i++) {
		lcd.setCursor(8, i); lcd.print("|");
		lcd.setCursor(10, i); lcd.print("|");
		delay(120);
	}
	delay(250);
}

void mostrarCopaASCII() {
	for (int i = 1; i <= 3; i++) {
		lcd.setCursor(0, i);
		lcd.print("                    ");
	}

	lcd.setCursor(0, 1);
	lcd.print("    /(------)");
	lcd.write(4);
	lcd.print("    ");

	lcd.setCursor(0, 2);
	lcd.print("    ");
	lcd.write(4);
	lcd.print("_) #1 (_/   ");

	lcd.setCursor(0, 3);
	lcd.print("      (____)       ");

	delay(2000);
}

void actualizarLCD()
{
	bool hayCambio = false;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0) {
				hayCambio = true;
			}
		}
	}

	lcd.setCursor(0, 0);
	lcd.print("===  [ BOARD ]   ===");

	if (hayCambio) {
		for (int i = 0; i < 3; i++) {
			lcd.setCursor(0, i + 1);
			lcd.print("      [");
			for (int j = 0; j < 3; j++) {
				bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
				if (esNueva) lcd.print(".");
				else if (tableroAnterior[i][j] == 0) lcd.print(" ");
				else if (tableroAnterior[i][j] == 1) lcd.print("X");
				else if (tableroAnterior[i][j] == 2) lcd.print("O");
				if (j < 2) lcd.print("|");
			}
			lcd.print("]       ");
			if (modoAutomatico) dibujarDecoracionTurnoAuto();
		}
		delay(150); 

		for (int i = 0; i < 3; i++) {
			lcd.setCursor(0, i + 1);
			lcd.print("      [");
			for (int j = 0; j < 3; j++) {
				bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
				if (esNueva) lcd.write(3);
				else if (tableroAnterior[i][j] == 0) lcd.print(" ");
				else if (tableroAnterior[i][j] == 1) lcd.print("X");
				else if (tableroAnterior[i][j] == 2) lcd.print("O");
				if (j < 2) lcd.print("|");
			}
			lcd.print("]       ");
			if (modoAutomatico) dibujarDecoracionTurnoAuto();
		}
		delay(150); 
	}

	lcd.setCursor(0, 0);
	lcd.print("===  [ BOARD ]   ===");
	for (int i = 0; i < 3; i++) {
		lcd.setCursor(0, i + 1);
		lcd.print("      [");
		for (int j = 0; j < 3; j++) {
			if (tablero[i][j] == 0) lcd.print(" ");
			else if (tablero[i][j] == 1) lcd.print("X");
			else if (tablero[i][j] == 2) lcd.print("O");
			if (j < 2) lcd.print("|");
			tableroAnterior[i][j] = tablero[i][j];
		}
		lcd.print("]       ");
		if (modoAutomatico) dibujarDecoracionTurnoAuto();
	}

	int ganador = comprobarGanador();
  
	if (ganador == 1 || ganador == 2) {
		delay(1200);
		mostrarCopaASCII();
		delay(1500);
	} 
	else if (ganador == 3) {
		delay(1500);
	}
}

int comprobarGanador()
{
	for (int i = 0; i < 3; i++)
	{
		if (tablero[i][0] != 0 && tablero[i][0] == tablero[i][1] && tablero[i][1] == tablero[i][2])
		{
			return tablero[i][0];
		}
	}

	for (int i = 0; i < 3; i++)
	{
		if (tablero[0][i] != 0 && tablero[0][i] == tablero[1][i] && tablero[1][i] == tablero[2][i])
		{
			return tablero[0][i];
		}
	}

	if (tablero[0][0] != 0 && tablero[0][0] == tablero[1][1] && tablero[1][1] == tablero[2][2])
	{
		return tablero[0][0];
	}

	if (tablero[0][2] != 0 && tablero[0][2] == tablero[1][1] && tablero[1][1] == tablero[2][0])
	{
		return tablero[0][2];
	}

	bool hayEspacioVacio = false;
	for (int f = 0; f < 3; f++)
	{
		for (int c = 0; c < 3; c++)
		{
			if (tablero[f][c] == 0)
			{
				hayEspacioVacio = true; 
				break;
			}
		}
	}

	if (!hayEspacioVacio)
	{
		return 3;
	}

	return 0;
}

void printBoardSerial()
{
	Serial.println("Matriz parseada:");
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Serial.print(tablero[i][j]);
			Serial.print(" ");
		}
		Serial.println();
	}
}

void vaciarTablero()
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			tablero[i][j] = 0; 
			tableroAnterior[i][j] = 0;
		}
	}
}
