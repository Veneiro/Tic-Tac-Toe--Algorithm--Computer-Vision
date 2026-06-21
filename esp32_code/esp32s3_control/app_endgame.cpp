#include "app_contracts.h"

/** @brief Gestiona el fin de partida: reproduce la canción de victoria/derrota/empate, muestra una animación
 *  de fuegos artificiales en el LCD con el texto del resultado en modo máquina de escribir y luego muestra la pantalla final "MATCH ENDED".
 *  @param ganador Código de resultado: 1 = gana el humano, 2 = gana el robot, 3 = empate; valores fuera de [1,3] no tienen efecto. */
void manejarFinDeJuego(int ganador)
{
  if (ganador >= 1 && ganador <= 3)
  {
    buzzerStop();  // para el boss battle (fase final) en todos los casos

    CancionId cancion = CANCION_NINGUNA;
    if      (ganador == 1) cancion = CANCION_HUMAN_WIN;
    else if (ganador == 2) cancion = CANCION_ROBOT_WIN;
    if (cancion != CANCION_NINGUNA) buzzerPlay(cancion, false);

    String linea1 = "RESULT";
    String linea2 = (ganador == 3) ? "ABSOLUTE DRAW"
                  : (ganador == 2) ? "ROBOT WINS!"
                  :                  "HUMAN WINS!";

    // Duración = duración real de la canción + 800 ms de margen.
    // Sin canción (empate) → 8 000 ms fijos.
    unsigned long duration = (cancion != CANCION_NINGUNA)
                           ? buzzerDuracion(cancion) + 800UL
                           : 8000UL;

    // linea1 empieza al 20% de la animación;
    // linea2 empieza justo al terminar linea1 + 200 ms de pausa.
    unsigned long tStart1 = duration / 5;
    unsigned long tStart2 = tStart1 + (unsigned long)(linea1.length() * 120UL) + 200UL;

    lcd.clear();
    unsigned long startTime = millis();

    // --- VARIABLES DE CONTROL DE FUEGOS ---
    static int fw_x = 10;
    static int fw_y_max = 1;

    char lastScreen[4][21] = {"                    ", "                    ", "                    ", "                    "};

    while (millis() - startTime < duration)
    {
      unsigned long t = millis() - startTime;
      char screen[4][21];
      for(int i = 0; i < 4; i++) strcpy(screen[i], "                    ");

      // --- LÓGICA DE FUEGO ARTIFICIAL (UNO A LA VEZ) ---
      // Ciclo de 1.5 segundos por cada cohete
      int fw_t = t % 1500; 

      // Al inicio de cada cohete, elegimos nueva posición X e Y aleatoria
      if (fw_t < 45) { 
        fw_x = random(2, 18);     // Horizontal: de columna 2 a 17
        fw_y_max = random(0, 3);  // Vertical: puede explotar en fila 0, 1 o 2
      }

      int x = fw_x;
      int y = fw_y_max; // Fila donde ocurre la explosión principal

      // Fases de la animación ajustadas a la altura Y elegida
      if (fw_t < 400) {
        // Fase 1: Cohete subiendo (solo si la explosión es arriba)
        if (y < 3) screen[3][x] = '|'; 
        if (y < 2 && fw_t > 200) screen[2][x] = '|';
      } 
      else if (fw_t < 600) {
        // Fase 2: El punto antes de estallar en su altura Y
        screen[y][x] = '*'; 
      } 
      else if (fw_t < 900) {
        // Fase 3: Explosión principal en la altura Y
        screen[y][x] = '+';
        if(y > 0) screen[y-1][x] = '|';
        if(y < 3) screen[y+1][x] = '|';
        if(x > 0) screen[y][x-1] = '-';
        if(x < 19) screen[y][x+1] = '-';
        // Diagonales de la explosión
        if(y > 0 && x > 0)  screen[y-1][x-1] = 4;  // Carácter personalizado: barra invertida
        if(y > 0 && x < 19) screen[y-1][x+1] = '/';
        if(y < 3 && x > 0)  screen[y+1][x-1] = '/';
        if(y < 3 && x < 19) screen[y+1][x+1] = 4;  // Carácter personalizado: barra invertida
      } 
      else if (fw_t < 1300) {
        // Fase 4: Chispas finales (disipación)
        if(y > 0 && x > 1)  screen[y-1][x-2] = '.';
        if(y < 3 && x < 18) screen[y+1][x+2] = '.';
        if(y < 2 && x > 1)  screen[y+2][x-2] = '.';
        screen[y][x] = '*';
      }

      // --- TEXTO: MÁQUINA DE ESCRIBIR (Sobreescribe el lienzo) ---
      int let1 = 0, let2 = 0;
      if (t > tStart1) {
        let1 = (t - tStart1) / 120;
        if (let1 > (int)linea1.length()) let1 = linea1.length();
      }
      if (t > tStart2) {
        let2 = (t - tStart2) / 120;
        if (let2 > (int)linea2.length()) let2 = linea2.length();
      }

      // Dibujar texto (siempre que no haya una chispa justo ahí, el texto manda)
      for (int i = 0; i < let1; i++) screen[1][2 + i] = linea1[i];
      for (int i = 0; i < let2; i++) screen[2][2 + i] = linea2[i];

      // --- CURSOR PARPADEANTE ---
      int cursorX = -1, cursorY = -1;
      bool showCursor = (t / 250) % 2 == 0;
      if (t > tStart1 && t <= tStart2 && let1 < (int)linea1.length()) {
        cursorX = 2 + let1; cursorY = 1;
      } else if (t > tStart2 && let2 < (int)linea2.length()) {
        cursorX = 2 + let2; cursorY = 2;
      }

      if (cursorX >= 0 && cursorX < 20 && cursorY >= 0 && cursorY < 4) {
         if (showCursor) screen[cursorY][cursorX] = (char)255;
         else screen[cursorY][cursorX] = ' ';
      }

      // --- DIBUJADO EFICIENTE ---
      for (int i = 0; i < 4; i++) {
        if (memcmp(screen[i], lastScreen[i], 20) != 0) {
          lcd.setCursor(0, i);
          for (int j = 0; j < 20; j++) {
            if (screen[i][j] == 4) {  // Código del carácter personalizado
              lcd.write(4);            // Escribe el carácter personalizado
            } else {
              lcd.print(screen[i][j]); // Escribe caracteres normales
            }
          }
          memcpy(lastScreen[i], screen[i], 21);
        }
      }
      delay(40); 
    }

    buzzerStop();

    // Mensaje final
    lcd.clear();
    lcd.setCursor(3, 1); lcd.print("MATCH ENDED");
    lcd.setCursor(4, 2); lcd.print("EXITING...");
    delay(2000);
  }
}


