#include "app_contracts.h"

void manejarFinDeJuego(int ganador)
{
	if (ganador >= 1 && ganador <= 3)
	{
		lcd.clear();
		unsigned long startTime = millis();
		unsigned long duration = 12000; 

		static int fw_x = 10;
		static int fw_y_max = 1;
    
		String linea1 = "MATCH ENDED";
		String linea2 = "";

		if (ganador == 3) {
			linea2 = "ABSOLUTE DRAW";
		} else {
			linea2 = "PLAYER " + String(ganador) + " WIN!";
		}

		char lastScreen[4][21] = {"                    ", "                    ", "                    ", "                    "};

		while (millis() - startTime < duration)
		{
			unsigned long t = millis() - startTime;
			char screen[4][21];
			for(int i = 0; i < 4; i++) strcpy(screen[i], "                    ");

			int fw_t = t % 1500; 

			if (fw_t < 45) { 
				fw_x = random(2, 18);
				fw_y_max = random(0, 3);
			}

			int x = fw_x;
			int y = fw_y_max;

			if (fw_t < 400) {
				if (y < 3) screen[3][x] = '|'; 
				if (y < 2 && fw_t > 200) screen[2][x] = '|';
			} 
			else if (fw_t < 600) {
				screen[y][x] = '*'; 
			} 
			else if (fw_t < 900) {
				screen[y][x] = '+';
				if(y > 0) screen[y-1][x] = '|';
				if(y < 3) screen[y+1][x] = '|';
				if(x > 0) screen[y][x-1] = '-';
				if(x < 19) screen[y][x+1] = '-';
				if(y > 0 && x > 0)  screen[y-1][x-1] = 4;
				if(y > 0 && x < 19) screen[y-1][x+1] = '/';
				if(y < 3 && x > 0)  screen[y+1][x-1] = '/';
				if(y < 3 && x < 19) screen[y+1][x+1] = 4;
			} 
			else if (fw_t < 1300) {
				if(y > 0 && x > 1)  screen[y-1][x-2] = '.';
				if(y < 3 && x < 18) screen[y+1][x+2] = '.';
				if(y < 2 && x > 1)  screen[y+2][x-2] = '.';
				screen[y][x] = '*';
			}

			int let1 = 0, let2 = 0;
			if (t > 2500) {
				let1 = (t - 2500) / 120;
				if (let1 > (int)linea1.length()) let1 = linea1.length();
			}
			if (t > 5000) {
				let2 = (t - 5000) / 120;
				if (let2 > (int)linea2.length()) let2 = linea2.length();
			}

			for (int i = 0; i < let1; i++) screen[1][2 + i] = linea1[i];
			for (int i = 0; i < let2; i++) screen[2][2 + i] = linea2[i];

			int cursorX = -1, cursorY = -1;
			bool showCursor = (t / 250) % 2 == 0;
			if (t > 2500 && t <= 5000 && let1 < (int)linea1.length()) {
				cursorX = 2 + let1; cursorY = 1;
			} else if (t > 5000 && let2 < (int)linea2.length()) {
				cursorX = 2 + let2; cursorY = 2;
			}

			if (cursorX >= 0 && cursorX < 20 && cursorY >= 0 && cursorY < 4) {
				 if (showCursor) screen[cursorY][cursorX] = (char)255;
				 else screen[cursorY][cursorX] = ' ';
			}

			for (int i = 0; i < 4; i++) {
				if (memcmp(screen[i], lastScreen[i], 20) != 0) {
					lcd.setCursor(0, i);
					for (int j = 0; j < 20; j++) {
						if (screen[i][j] == 4) {
							lcd.write(4);
						} else {
							lcd.print(screen[i][j]);
						}
					}
					memcpy(lastScreen[i], screen[i], 21);
				}
			}
			delay(40); 
		}

		lcd.clear();
		lcd.setCursor(3, 1); lcd.print("MATCH ENDED");
		lcd.setCursor(4, 2); lcd.print("EXITING...");
		delay(2000);
	}
}
