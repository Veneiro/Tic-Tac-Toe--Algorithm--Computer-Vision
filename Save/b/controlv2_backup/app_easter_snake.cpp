#include "app_contracts.h"

void jugarSnake() {
	lcd.clear();
	unsigned long startAnim = millis();
	String txt1 = "--- SNAKE ---";
	String txt2 = "EASTER EGG FOUND";
  
	char lastScreen[4][21] = {"                    ", "                    ", "                    ", "                    "};

	while(millis() - startAnim < 7000) {
		unsigned long t = millis() - startAnim;
		char screen[4][21];
		for(int i=0; i<4; i++) strcpy(screen[i], "                    ");

		int frame = (t / 150) % 4;
		char chars1[] = {'|', '/', '-', 4};
		char chars2[] = {'+', 'x', '*', 'o'};
    
		for(int r = 0; r < 4; r++) {
			 for(int c = 0; c < 20; c++) {
					 int dist = abs(c - 9) + abs(r - 1);
					 if (dist == (t / 200) % 12) screen[r][c] = chars1[frame];
					 else if (dist == ((t / 200) + 4) % 12) screen[r][c] = chars2[frame];
			 }
		}

		int l1 = (t > 2500) ? (t - 2500) / 120 : 0;
		if (l1 > (int)txt1.length()) l1 = txt1.length();
		int l2 = (t > 4500) ? (t - 4500) / 120 : 0;
		if (l2 > (int)txt2.length()) l2 = txt2.length();

		for(int i=0; i<l1; i++) screen[1][3+i] = txt1[i];
		for(int i=0; i<l2; i++) screen[2][2+i] = txt2[i];

		bool cur = (t / 250) % 2 == 0;
		if (t > 2500 && t < 4500 && l1 < (int)txt1.length() && cur) screen[1][3+l1] = (char)255;
		if (t > 4500 && l2 < (int)txt2.length() && cur) screen[2][2+l2] = (char)255;

		for(int i=0; i<4; i++) {
			if (strcmp(screen[i], lastScreen[i]) != 0) {
				lcd.setCursor(0, i);
				for (int j = 0; j < 20; j++) {
					if (screen[i][j] == 4) {
						lcd.write(4);
					} else if (screen[i][j] == (char)255) {
						lcd.write(255);
					} else {
						lcd.print(screen[i][j]);
					}
				}
				strcpy(lastScreen[i], screen[i]);
			}
		}
		delay(40);
	}
	lcd.clear();

	byte snakeHeadRight[8] = {0b00000, 0b01100, 0b11110, 0b11111, 0b11110, 0b01100, 0b00000, 0b00000};
	byte snakeHeadLeft[8]  = {0b00000, 0b00110, 0b01111, 0b11111, 0b01111, 0b00110, 0b00000, 0b00000};
	byte snakeHeadUp[8]    = {0b00000, 0b00100, 0b01110, 0b11111, 0b10101, 0b00100, 0b00000, 0b00000};
	byte snakeHeadDown[8]  = {0b00000, 0b00100, 0b10101, 0b11111, 0b01110, 0b00100, 0b00000, 0b00000};
	byte snakeBody[8]      = {0b00000, 0b01110, 0b11111, 0b11111, 0b11111, 0b01110, 0b00000, 0b00000};
	byte snakeApple[8]     = {0b00100, 0b01010, 0b01110, 0b11111, 0b11111, 0b01110, 0b00100, 0b00000};

	lcd.createChar(0, snakeHeadRight);
	lcd.createChar(1, snakeHeadLeft);
	lcd.createChar(2, snakeHeadUp);
	lcd.createChar(3, snakeHeadDown);
	lcd.createChar(4, snakeBody);
	lcd.createChar(5, snakeApple);

	bool reiniciar = true;

	while (reiniciar) {
		reiniciar = false;
		int snakeX[80], snakeY[80];
		int snakeLen = 3;
		int dir = 1;

		for(int i = 0; i < snakeLen; i++) {
			snakeX[i] = 10 - i;
			snakeY[i] = 2;
		}
    
		int foodX = random(0, 20);
		int foodY = random(0, 4);

		bool playing = true;
		int score = 0;
		bool salidaForzada = false;
		bool antMenuSnake = digitalRead(pinMenu);

		auto dibujarSnakeCompleta = [&]() {
			lcd.clear();
			lcd.setCursor(foodX, foodY);
			lcd.write(5);

			for (int i = snakeLen - 1; i > 0; i--) {
				lcd.setCursor(snakeX[i], snakeY[i]);
				lcd.write(4);
			}

			byte headChar = 0;
			if (dir == 0) headChar = 2;
			else if (dir == 1) headChar = 0;
			else if (dir == 2) headChar = 3;
			else if (dir == 3) headChar = 1;
			lcd.setCursor(snakeX[0], snakeY[0]);
			lcd.write(headChar);
		};

		dibujarSnakeCompleta();

		while(playing) {
			server.handleClient();

			bool lecturaMenuSnake = digitalRead(pinMenu);
			if (lecturaMenuSnake == HIGH && antMenuSnake == LOW) {
				esperarLiberacionBoton(pinMenu);
				bool salirSnake = abrirMenuPausa();
				if (salirSnake) {
					salidaForzada = true;
					break;
				}
				dibujarSnakeCompleta();
			}
			antMenuSnake = lecturaMenuSnake;

			unsigned long tiempoFrame = millis();
			while (millis() - tiempoFrame < 250) { 
				server.handleClient(); 
        
				int joyX = analogRead(pinJoyX);
				int joyY = analogRead(pinJoyY);
				int difX = abs(joyX - 2048);
				int difY = abs(joyY - 2048);

				if (difX > difY && difX > 800) {
					if (joyX < 1200 && dir != 1) dir = 3;      
					else if (joyX > 2800 && dir != 3) dir = 1; 
				} 
				else if (difY > difX && difY > 800) {
					if (joyY < 1200 && dir != 2) dir = 0;      
					else if (joyY > 2800 && dir != 0) dir = 2; 
				}
				delay(5); 
			}

			lcd.setCursor(snakeX[snakeLen-1], snakeY[snakeLen-1]);
			lcd.print(" ");

			int oldHeadX = snakeX[0];
			int oldHeadY = snakeY[0];

			for(int i = snakeLen - 1; i > 0; i--) {
				snakeX[i] = snakeX[i-1];
				snakeY[i] = snakeY[i-1];
			}

			if(dir == 0) snakeY[0]--;
			else if(dir == 1) snakeX[0]++;
			else if(dir == 2) snakeY[0]++;
			else if(dir == 3) snakeX[0]--;

			if(snakeX[0] < 0) snakeX[0] = 19;
			else if(snakeX[0] >= 20) snakeX[0] = 0;
			if(snakeY[0] < 0) snakeY[0] = 3;
			else if(snakeY[0] >= 4) snakeY[0] = 0;

			for(int i = 1; i < snakeLen; i++) {
				if(snakeX[0] == snakeX[i] && snakeY[0] == snakeY[i]) playing = false;
			}
      
			if(!playing) break;

			if(snakeX[0] == foodX && snakeY[0] == foodY) {
				if(snakeLen < 80) snakeLen++;
				score += 10;
        
				bool valid = false;
				while(!valid) {
					foodX = random(0, 20);
					foodY = random(0, 4);
					valid = true;
					for(int i = 0; i < snakeLen; i++) {
						if(foodX == snakeX[i] && foodY == snakeY[i]) valid = false;
					}
				}
			}

			lcd.setCursor(foodX, foodY); lcd.write(5);
			lcd.setCursor(oldHeadX, oldHeadY); lcd.write(4);

			byte headChar = 0;
			if (dir == 0) headChar = 2;
			else if (dir == 1) headChar = 0;
			else if (dir == 2) headChar = 3;
			else if (dir == 3) headChar = 1;
			lcd.setCursor(snakeX[0], snakeY[0]); lcd.write(headChar);
		}

		if (salidaForzada) break;

		delay(500);

		int hX = snakeX[0];
		int hY = snakeY[0];
    
		for (int radius = 0; radius <= 22; radius++) {
			for (int r = 0; r < 4; r++) {
				for (int c = 0; c < 20; c++) {
					int dist = abs(c - hX) + abs(r - hY);
					if (dist == radius) {
						lcd.setCursor(c, r);
						lcd.write(255);
					}
				}
			}
			delay(40);
		}
    
		delay(200);
		transicionBarrido();

		lcd.setCursor(5, 0); lcd.print("GAME OVER");
    
		char bufScore[21];
		snprintf(bufScore, sizeof(bufScore), "   SCORE: %04d   ", score);
		lcd.setCursor(0, 1); lcd.print(bufScore);

		lcd.setCursor(0, 2); lcd.print(" START: Exit");
		lcd.setCursor(0, 3); lcd.print(" MENU:  Retry");

		while (true) {
			server.handleClient();
			if (digitalRead(pinStart) == HIGH) {
				esperarLiberacionBoton(pinStart);
				reiniciar = false;
				break;
			}
			if (digitalRead(pinMenu) == HIGH) {
				esperarLiberacionBoton(pinMenu);
				reiniciar = true;
				break;
			}
			delay(20);
		}
	}

	cargarCaracteresBase();
}
