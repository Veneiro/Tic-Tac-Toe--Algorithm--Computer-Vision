#include "app_contracts.h"

void esperarSeleccionMenu()
{
	bool ultimoEstadoSwitch = !digitalRead(pinSwitch); 
	char ultimaFila3[21] = "";

	lcd.clear();

	while (digitalRead(pinStart) == LOW)
	{
		server.handleClient(); 
		unsigned long t = millis();

		bool frameEngranaje = (t / 250) % 2 == 0; 
		lcd.setCursor(3, 0);
		lcd.write(frameEngranaje ? 5 : 6); 
		lcd.print(" MAIN  MENU ");
		lcd.write(frameEngranaje ? 6 : 5); 

		bool lecturaSwitch = digitalRead(pinSwitch);
		if (lecturaSwitch != ultimoEstadoSwitch)
		{
			if (lecturaSwitch == LOW) { 
				lcd.setCursor(0, 1); lcd.print(" --[    AUTO    ]-- ");
				lcd.setCursor(9, 2); lcd.write(0); 
				modoAutomatico = true;  
			} else { 
				lcd.setCursor(0, 1); lcd.print(" --[   MANUAL   ]-- ");
				lcd.setCursor(9, 2); lcd.write(2); 
				modoAutomatico = false; 
			}
			ultimoEstadoSwitch = lecturaSwitch;
		}

		char filaActual3[21] = "                    ";
		bool flechasDentro = (t / 400) % 2 == 0;
		if (flechasDentro) {
			filaActual3[2] = '>'; filaActual3[3] = '>';
			filaActual3[15] = '<'; filaActual3[16] = '<';
		} else {
			filaActual3[1] = '>'; filaActual3[2] = '>';
			filaActual3[16] = '<'; filaActual3[17] = '<';
		}

		const char* txt = "PRESS START";
		int longitudTexto = 11;
		int cicloText = t % 3000;
    
		int startIdx = 0;
		int endIdx = 0;

		if (cicloText <= 1200) {
			startIdx = 0;
			endIdx = (cicloText * longitudTexto) / 1200;
		} 
		else if (cicloText <= 1800) {
			startIdx = 0;
			endIdx = longitudTexto;
		} 
		else {
			startIdx = ((cicloText - 1800) * longitudTexto) / 1200;
			endIdx = longitudTexto;
		}
    
		for (int i = 0; i < longitudTexto; i++) {
			if (i >= startIdx && i < endIdx) {
				filaActual3[4 + i] = txt[i]; 
			}
		}

		if (strcmp(filaActual3, ultimaFila3) != 0) {
			lcd.setCursor(0, 3);
			lcd.print(filaActual3);
			strcpy(ultimaFila3, filaActual3); 
		}

		delay(20); 
	}
  
	if (digitalRead(pinMenu) == HIGH && digitalRead(pinJoyButton) == LOW) {
		modoEspecialTipo = 2;
	} else if (digitalRead(pinMenu) == HIGH) {
		modoEspecialTipo = 1;
	} else {
		modoEspecialTipo = 0;
	}

	esperarLiberacionBoton(pinStart);
  
	if (modoEspecialTipo != 0) {
		esperarLiberacionBoton(pinMenu);
		if (modoEspecialTipo == 2) {
			while (digitalRead(pinJoyButton) == LOW) {
				server.handleClient();
				delay(10);
			}
			delay(40);
		}
	} else {
		confirmarInicio();
	}
}

void confirmarInicio()
{
	lcd.clear();

	if (modoAutomatico) 
	{
		lcd.setCursor(0, 0); lcd.print("     ___            ");
		lcd.setCursor(0, 1); lcd.print("    [off]           ");
		lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4); lcd.print("          ");
		lcd.setCursor(0, 3); lcd.print("   ==| |==          ");
		delay(500);
    
		lcd.setCursor(0, 0); lcd.print("     _|_            ");
		lcd.setCursor(0, 1); lcd.print("    [-_-]           ");
		lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4); lcd.print("          ");
		delay(400);

		lcd.setCursor(0, 0); lcd.print("     "); lcd.write(4); lcd.print("|/            ");
		lcd.setCursor(0, 1); lcd.print("    [O_O]   AUTO    ");
		lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4); lcd.print("  MODE    ");
		delay(400);
    
		for(int i = 0; i < 3; i++) {
			 lcd.setCursor(5, 1); lcd.print(">_<"); delay(150);
			 lcd.setCursor(5, 1); lcd.print("O_O"); delay(150);
		}
		delay(400);
	} 
	else 
	{
		lcd.setCursor(3, 0); 
		lcd.print("MANUAL MODE OK");
    
		lcd.setCursor(0, 1); lcd.print("X:[             ]");
		lcd.setCursor(0, 2); lcd.print("Y:[             ]");
		lcd.setCursor(1, 3); lcd.print(" Calibrating Axes ");

		for (float t = 0; t <= 6.28; t += 0.35) 
		{
			int posX = 9 + 6 * sin(t); 
			int posY = 9 + 6 * cos(t);

			lcd.setCursor(3, 1);
			for(int i = 3; i <= 15; i++) {
				if (i == posX) lcd.write(255);
				else lcd.print("-");
			}

			lcd.setCursor(3, 2);
			for(int i = 3; i <= 15; i++) {
				if (i == posY) lcd.write(255); 
				else lcd.print("-");
			}

			delay(60); 
		}

		lcd.setCursor(1, 3); 
		lcd.print("   SYSTEM READY!   ");
		delay(600);
	}
}
