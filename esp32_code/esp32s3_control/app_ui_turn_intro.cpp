#include "app_contracts.h"

void mostrarPantallaTurnoInicial(bool empiezaMaquina)
{
  const String lineaTurno = empiezaMaquina ? "ROBOT'S TURN" : "HUMAN'S TURN";
  const int xTurno = (20 - (int)lineaTurno.length()) / 2;

  // Estados: 0 = cayendo, 1..4 = impacto/onda.
  const int gotas = 8;
  byte estado[gotas];
  int gotaX[gotas];
  int gotaY[gotas];
  unsigned long proximoPaso[gotas];

  for (int i = 0; i < gotas; i++)
  {
    estado[i] = 0;
    gotaX[i] = random(0, 20);
    gotaY[i] = random(-8, 1);
    proximoPaso[i] = millis() + (unsigned long)random(40, 180);
  }

  char lastScreen[4][21] = {
    "                    ",
    "                    ",
    "                    ",
    "                    "
  };

  unsigned long inicio = millis();
  unsigned long duracion = 5200;
  unsigned long ultimoTickGotas = 0;

  while (millis() - inicio < duracion)
  {
    server.handleClient();
    unsigned long t = millis() - inicio;

    // Avance temporal de gotas + ondas con tiempos aleatorios.
    if (millis() - ultimoTickGotas >= 25)
    {
      for (int i = 0; i < gotas; i++)
      {
        if (millis() < proximoPaso[i]) continue;

        if (estado[i] == 0)
        {
          gotaY[i]++;
          if (gotaY[i] >= 3)
          {
            gotaY[i] = 3;
            estado[i] = 1;
            proximoPaso[i] = millis() + (unsigned long)random(65, 120);
          }
          else
          {
            proximoPaso[i] = millis() + (unsigned long)random(55, 125);
          }
        }
        else
        {
          estado[i]++;
          if (estado[i] > 4)
          {
            estado[i] = 0;
            gotaX[i] = random(0, 20);
            gotaY[i] = random(-10, -2);
            proximoPaso[i] = millis() + (unsigned long)random(120, 420);
          }
          else
          {
            proximoPaso[i] = millis() + (unsigned long)random(70, 130);
          }
        }
      }
      ultimoTickGotas = millis();
    }

    char screen[4][21];
    for (int r = 0; r < 4; r++) strcpy(screen[r], "                    ");

    // Lluvia ASCII con impacto y ondas (estilo rain-drops).
    for (int i = 0; i < gotas; i++)
    {
      int x = gotaX[i];

      if (estado[i] == 0)
      {
        int y = gotaY[i];
        if (y >= 0 && y < 4 && x >= 0 && x < 20) screen[y][x] = '|';
        if (y - 1 >= 0 && y - 1 < 4 && x >= 0 && x < 20) screen[y - 1][x] = '.';
      }
      else if (estado[i] == 1)
      {
        if (x >= 0 && x < 20) screen[3][x] = 'o';
      }
      else if (estado[i] == 2)
      {
        if (x - 1 >= 0) screen[3][x - 1] = '(';
        if (x + 1 < 20) screen[3][x + 1] = ')';
        if (x >= 0 && x < 20) screen[2][x] = '.';
      }
      else if (estado[i] == 3)
      {
        if (x - 2 >= 0) screen[3][x - 2] = '(';
        if (x + 2 < 20) screen[3][x + 2] = ')';
        if (x - 1 >= 0) screen[3][x - 1] = '-';
        if (x + 1 < 20) screen[3][x + 1] = '-';
        if (x >= 0 && x < 20) screen[3][x] = '_';
      }
      else if (estado[i] == 4)
      {
        if (x - 3 >= 0) screen[3][x - 3] = '.';
        if (x + 3 < 20) screen[3][x + 3] = '.';
        if (x - 2 >= 0) screen[3][x - 2] = '-';
        if (x + 2 < 20) screen[3][x + 2] = '-';
      }
    }

    int letrasTurno = 0;
    if (t > 700)
    {
      letrasTurno = (t - 700) / 90;
      if (letrasTurno > (int)lineaTurno.length()) letrasTurno = lineaTurno.length();
    }

    for (int i = 0; i < letrasTurno; i++)
    {
      int x = xTurno + i;
      if (x >= 0 && x < 20) screen[1][x] = lineaTurno[i];
    }

    // Cursor parpadeante durante la escritura.
    bool showCursor = (t / 220) % 2 == 0;
    if (showCursor)
    {
      if (t > 700 && letrasTurno < (int)lineaTurno.length())
      {
        int cx = xTurno + letrasTurno;
        if (cx >= 0 && cx < 20) screen[1][cx] = (char)255;
      }
    }

    for (int r = 0; r < 4; r++)
    {
      if (memcmp(screen[r], lastScreen[r], 20) != 0)
      {
        lcd.setCursor(0, r);
        for (int c = 0; c < 20; c++)
        {
          if (screen[r][c] == (char)255) lcd.write(255);
          else lcd.print(screen[r][c]);
        }
        memcpy(lastScreen[r], screen[r], 21);
      }
    }

    delay(40);
  }

  transicionBarrido();
  lcd.clear();
}


