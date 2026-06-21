#include "app_contracts.h"

/** @brief Minijuego Easter-egg Pac-Man: bucle de juego completo con renderizado en LCD, entrada del joystick,
 *  IA de fantasmas en modo dispersión/persecución, píldoras de poder, bonificaciones de fruta, puntuación y pantalla de reintento.
 *  Restaura los caracteres base del LCD y ejecuta una transición de barrido al salir. */
void jugarPacman() {
  lcd.clear();

  // --- Intro estilo Snake pero con PACMAN ---
  unsigned long startAnim = millis();
  String txt1 = "--- PACMAN ---";
  String txt2 = "EASTER EGG FOUND";
  char lastScreen[4][21] = {
    "                    ",
    "                    ",
    "                    ",
    "                    "
  };

  while (millis() - startAnim < 5200) {
    server.handleClient();
    unsigned long t = millis() - startAnim;
    char screen[4][21];
    for (int i = 0; i < 4; i++) strcpy(screen[i], "                    ");

    int frame = (t / 150) % 4;
    char chars1[] = {'|', '/', '-', 4};
    char chars2[] = {'+', 'x', '*', 'o'};

    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 20; c++) {
        int dist = abs(c - 9) + abs(r - 1);
        if (dist == (t / 200) % 12) screen[r][c] = chars1[frame];
        else if (dist == ((t / 200) + 4) % 12) screen[r][c] = chars2[frame];
      }
    }

    int l1 = (t > 1200) ? (t - 1200) / 110 : 0;
    if (l1 > (int)txt1.length()) l1 = txt1.length();
    int l2 = (t > 2800) ? (t - 2800) / 95 : 0;
    if (l2 > (int)txt2.length()) l2 = txt2.length();

    for (int i = 0; i < l1; i++) screen[1][3 + i] = txt1[i];
    for (int i = 0; i < l2; i++) screen[2][2 + i] = txt2[i];

    bool cur = (t / 250) % 2 == 0;
    if (t > 1200 && t < 2800 && l1 < (int)txt1.length() && cur) screen[1][3 + l1] = (char)255;
    if (t > 2800 && l2 < (int)txt2.length() && cur) screen[2][2 + l2] = (char)255;

    for (int i = 0; i < 4; i++) {
      if (strcmp(screen[i], lastScreen[i]) != 0) {
        lcd.setCursor(0, i);
        for (int j = 0; j < 20; j++) {
          if (screen[i][j] == 4) lcd.write(4);
          else if (screen[i][j] == (char)255) lcd.write(255);
          else lcd.print(screen[i][j]);
        }
        strcpy(lastScreen[i], screen[i]);
      }
    }
    delay(40);
  }

  // --- Caracteres personalizados temporales para Pacman ---
  byte charPacRight[8]  = {0b00000, 0b01110, 0b11011, 0b11100, 0b11100, 0b11011, 0b01110, 0b00000};
  byte charPacLeft[8]   = {0b00000, 0b01110, 0b11011, 0b00111, 0b00111, 0b11011, 0b01110, 0b00000};
  byte charPacClosed[8] = {0b00000, 0b01110, 0b11111, 0b11111, 0b11111, 0b11111, 0b01110, 0b00000};
  byte charGhost[8]     = {0b00000, 0b01110, 0b11111, 0b10101, 0b11111, 0b11111, 0b10101, 0b00000};
  byte charGhostFear[8] = {0b00000, 0b01110, 0b11111, 0b10001, 0b11111, 0b10101, 0b11111, 0b00000};
  byte charLife[8]      = {0b00000, 0b01110, 0b11111, 0b11100, 0b11100, 0b11111, 0b01110, 0b00000};
  byte charFruit[8]     = {0b00000, 0b00100, 0b01110, 0b11111, 0b11111, 0b01110, 0b00100, 0b00000};
  byte charEyes[8]      = {0b00000, 0b00000, 0b11011, 0b11011, 0b00000, 0b00000, 0b00000, 0b00000};

  lcd.createChar(0, charPacRight);
  lcd.createChar(1, charPacLeft);
  lcd.createChar(2, charPacClosed);
  lcd.createChar(3, charGhost);
  lcd.createChar(4, charGhostFear);
  lcd.createChar(5, charLife);
  lcd.createChar(6, charFruit);
  lcd.createChar(7, charEyes);

  bool reiniciar = true;
  while (reiniciar) {
    reiniciar = false;

    const int W = 20;
    const int H = 3;
    byte mapa[H][W]; // 0 vacio, 1 pellet, 2 power pellet, 3 pared

    // Mapa más jugable (sin salida bloqueada al spawn).
    const char *plantilla[H] = {
      "....................",
      ".##....##..##....##.",
      "...................."
    };

    const int G = 4;
    int gStartX[G] = {10, 9, 6, 13};
    int gStartY[G] = {1, 1, 1, 1};
    int gx[G], gy[G], gDir[G], gEstado[G];
    // Estado fantasma: 0 normal, 1 frightened, 2 eaten

    int pacX = 1, pacY = 2;
    int dirPac = 1, nextDirPac = 1; // 0 up, 1 right, 2 down, 3 left

    int score = 0;
    int vidas = 3;
    int nivel = 1;
    int comboFantasmas = 200;
    bool salidaForzada = false;
    bool antMenuPacman = digitalRead(pinMenu);
    bool extraVidaDada = false;

    unsigned long miedoHasta = 0;
    unsigned long lastPacMove = 0;
    unsigned long lastGhostMove = 0;
    unsigned long faseStart = millis();
    unsigned long invulnerableHasta = 0;
    unsigned long frutaHasta = 0;
    bool frutaActiva = false;
    int frutaX = 10;
    int frutaY = 1;

    unsigned long ghostReleaseAt[G];
    unsigned long faseCambio = 0;
    int faseModo = 0; // 0 scatter, 1 chase
    int fasePaso = 0;

    auto iniciarNivel = [&]() {
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          mapa[y][x] = (plantilla[y][x] == '#') ? 3 : 1;
        }
      }

      mapa[0][0] = 2;
      mapa[0][19] = 2;
      mapa[2][0] = 2;
      mapa[2][19] = 2;

      pacX = 1;
      pacY = 2;
      dirPac = 1;
      nextDirPac = 1;

      for (int i = 0; i < G; i++) {
        gx[i] = gStartX[i];
        gy[i] = gStartY[i];
        gDir[i] = (i % 2 == 0) ? 1 : 3;
        gEstado[i] = 0;
      }

      // Salida escalonada como en original.
      unsigned long ahora = millis();
      ghostReleaseAt[0] = ahora + 500;
      ghostReleaseAt[1] = ahora + 2200;
      ghostReleaseAt[2] = ahora + 4200;
      ghostReleaseAt[3] = ahora + 6200;

      faseModo = 0;
      fasePaso = 0;
      faseCambio = ahora + 7000;

      comboFantasmas = 200;
      miedoHasta = 0;
      frutaActiva = false;
      frutaHasta = 0;
      invulnerableHasta = ahora + 1800;
      faseStart = ahora;
      lastPacMove = ahora;
      lastGhostMove = ahora;
    };

    auto puedeMover = [&](int x, int y, int dir) {
      int nx = x, ny = y;
      if (dir == 0) ny--;
      if (dir == 1) nx++;
      if (dir == 2) ny++;
      if (dir == 3) nx--;

      if (nx < 0) nx = W - 1;
      if (nx >= W) nx = 0;
      if (ny < 0 || ny >= H) return false;
      return mapa[ny][nx] != 3;
    };

    auto moverPos = [&](int &x, int &y, int dir) {
      if (dir == 0) y--;
      if (dir == 1) x++;
      if (dir == 2) y++;
      if (dir == 3) x--;
      if (x < 0) x = W - 1;
      if (x >= W) x = 0;
    };

    auto pelletsRestantes = [&]() {
      int p = 0;
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          if (mapa[y][x] == 1 || mapa[y][x] == 2) p++;
        }
      }
      return p;
    };

    auto dirOpuesta = [&](int d) {
      return (d + 2) % 4;
    };

    auto distTaxi = [&](int x1, int y1, int x2, int y2) {
      int dx = abs(x1 - x2);
      dx = min(dx, W - dx);
      int dy = abs(y1 - y2);
      return dx + dy;
    };

    auto targetScatterX = [&](int i) {
      if (i == 0) return 19;
      if (i == 1) return 0;
      if (i == 2) return 19;
      return 0;
    };

    auto targetScatterY = [&](int i) {
      if (i == 0) return 0;
      if (i == 1) return 0;
      if (i == 2) return 2;
      return 2;
    };

    auto escogerDirFantasma = [&](int i, bool frightened) {
      int opciones[4];
      int n = 0;
      for (int d = 0; d < 4; d++) {
        if (!puedeMover(gx[i], gy[i], d)) continue;
        if (d == dirOpuesta(gDir[i])) continue; // evita giros bruscos salvo atajo
        opciones[n++] = d;
      }
      if (n == 0) {
        for (int d = 0; d < 4; d++) {
          if (puedeMover(gx[i], gy[i], d)) opciones[n++] = d;
        }
      }
      if (n == 0) return gDir[i];

      if (frightened) {
        return opciones[random(0, n)];
      }

      int tx = pacX;
      int ty = pacY;

      // Comportamientos inspirados en los originales.
      if (gEstado[i] == 2) {
        tx = gStartX[i];
        ty = gStartY[i];
      } else if (faseModo == 0) {
        tx = targetScatterX(i);
        ty = targetScatterY(i);
      } else {
        if (i == 0) {
          // Blinky: directo a Pac-Man.
          tx = pacX;
          ty = pacY;
        } else if (i == 1) {
          // Pinky: 3 celdas por delante.
          tx = pacX;
          ty = pacY;
          for (int k = 0; k < 3; k++) {
            if (dirPac == 0) ty--;
            if (dirPac == 1) tx++;
            if (dirPac == 2) ty++;
            if (dirPac == 3) tx--;
            if (tx < 0) tx = W - 1;
            if (tx >= W) tx = 0;
          }
        } else if (i == 2) {
          // Inky simplificado: vector entre Blinky y Pac-Man adelantado.
          int ax = pacX, ay = pacY;
          if (dirPac == 1) ax = (ax + 2) % W;
          if (dirPac == 3) ax = (ax - 2 + W) % W;
          if (dirPac == 0) ay = max(0, ay - 2);
          if (dirPac == 2) ay = min(H - 1, ay + 2);
          tx = (ax * 2 - gx[0] + W * 2) % W;
          ty = constrain(ay * 2 - gy[0], 0, H - 1);
        } else {
          // Clyde: persigue lejos, se dispersa cerca.
          if (distTaxi(gx[i], gy[i], pacX, pacY) > 6) {
            tx = pacX;
            ty = pacY;
          } else {
            tx = targetScatterX(i);
            ty = targetScatterY(i);
          }
        }
      }

      int bestD = opciones[0];
      int bestV = 9999;
      for (int k = 0; k < n; k++) {
        int d = opciones[k];
        int nx = gx[i], ny = gy[i];
        if (d == 0) ny--;
        if (d == 1) nx++;
        if (d == 2) ny++;
        if (d == 3) nx--;
        if (nx < 0) nx = W - 1;
        if (nx >= W) nx = 0;
        int v = distTaxi(nx, ny, tx, ty);
        if (v < bestV) {
          bestV = v;
          bestD = d;
        }
      }
      return bestD;
    };

    auto comprobarColisiones = [&](bool &muerto) {
      unsigned long ahora = millis();
      bool frightened = ahora < miedoHasta;
      for (int i = 0; i < G; i++) {
        if (gx[i] == pacX && gy[i] == pacY) {
          if (gEstado[i] == 2) continue;
          if (frightened) {
            gEstado[i] = 2; // comido
            score += comboFantasmas;
            comboFantasmas = min(comboFantasmas * 2, 1600);
          } else if (ahora > invulnerableHasta) {
            muerto = true;
            return;
          }
        }
      }
    };

    iniciarNivel();

    bool jugando = true;
    while (jugando) {
      server.handleClient();

      bool lecturaMenuPacman = digitalRead(pinMenu);
      if (lecturaMenuPacman == HIGH && antMenuPacman == LOW) {
        esperarLiberacionBoton(pinMenu);
        bool salirPacman = abrirMenuPausa();
        if (salirPacman) {
          salidaForzada = true;
          break;
        }
      }
      antMenuPacman = lecturaMenuPacman;

      unsigned long ahora = millis();

      // Ciclo scatter/chase clásico simplificado.
      if (ahora > faseCambio) {
        if (fasePaso < 7) {
          faseModo = (faseModo == 0) ? 1 : 0;
          fasePaso++;
          if (faseModo == 0) faseCambio = ahora + 7000;
          else faseCambio = ahora + 20000;
        } else {
          faseModo = 1;
          faseCambio = ahora + 60000;
        }
      }

      int joyX = analogRead(pinJoyX);
      int joyY = analogRead(pinJoyY);
      int difX = abs(joyX - 2048);
      int difY = abs(joyY - 2048);
      if (difX > difY && difX > 800) {
        if (joyX < 1200) nextDirPac = 3;
        else if (joyX > 2800) nextDirPac = 1;
      } else if (difY > difX && difY > 800) {
        if (joyY < 1200) nextDirPac = 0;
        else if (joyY > 2800) nextDirPac = 2;
      }

      int velPac = max(85, 165 - (nivel - 1) * 7);
      int velGhost = max(100, 205 - (nivel - 1) * 8);
      bool frightened = ahora < miedoHasta;
      if (frightened) velGhost += 35;

      // Fruta bonus (dos apariciones por nivel según pellets restantes).
      int pellets = pelletsRestantes();
      if (!frutaActiva && pellets < 30 && frutaHasta == 0) {
        frutaActiva = true;
        frutaHasta = ahora + 6000;
      } else if (!frutaActiva && pellets < 12 && frutaHasta != 0) {
        frutaActiva = true;
        frutaHasta = ahora + 5000;
      }
      if (frutaActiva && ahora > frutaHasta) frutaActiva = false;

      // READY inicial como en clásico.
      if (ahora - faseStart < 1300) {
        lcd.setCursor(0, 0); lcd.print("READY!      PACMAN ");
      }

      if (ahora - lastPacMove >= (unsigned long)velPac && ahora - faseStart >= 1300) {
        lastPacMove = ahora;

        if (puedeMover(pacX, pacY, nextDirPac)) dirPac = nextDirPac;
        if (puedeMover(pacX, pacY, dirPac)) moverPos(pacX, pacY, dirPac);

        if (mapa[pacY][pacX] == 1) {
          mapa[pacY][pacX] = 0;
          score += 10;
        } else if (mapa[pacY][pacX] == 2) {
          mapa[pacY][pacX] = 0;
          score += 50;
          miedoHasta = ahora + max(2600, 6800 - (nivel - 1) * 420);
          comboFantasmas = 200;
          for (int i = 0; i < G; i++) {
            if (gEstado[i] == 0) gEstado[i] = 1;
          }
        }

        if (frutaActiva && pacX == frutaX && pacY == frutaY) {
          score += 100 + nivel * 20;
          frutaActiva = false;
        }

        if (!extraVidaDada && score >= 10000) {
          vidas++;
          extraVidaDada = true;
        }

        bool muerto = false;
        comprobarColisiones(muerto);
        if (muerto) {
          vidas--;
          if (vidas <= 0) {
            jugando = false;
          } else {
            iniciarNivel();
            delay(350);
          }
        }

        if (pellets == 0) {
          nivel++;
          score += 300;
          iniciarNivel();
        }
      }

      if (ahora - lastGhostMove >= (unsigned long)velGhost && ahora - faseStart >= 1300) {
        lastGhostMove = ahora;

        for (int i = 0; i < G; i++) {
          if (ahora < ghostReleaseAt[i]) continue;

          bool fr = (gEstado[i] == 1) && (ahora < miedoHasta);
          int dirElegida = escogerDirFantasma(i, fr);
          gDir[i] = dirElegida;
          moverPos(gx[i], gy[i], gDir[i]);

          if (gEstado[i] == 2 && gx[i] == gStartX[i] && gy[i] == gStartY[i]) {
            gEstado[i] = 0;
          }
        }

        if (ahora > miedoHasta) {
          for (int i = 0; i < G; i++) {
            if (gEstado[i] == 1) gEstado[i] = 0;
          }
        }

        bool muerto = false;
        comprobarColisiones(muerto);
        if (muerto) {
          vidas--;
          if (vidas <= 0) {
            jugando = false;
          } else {
            iniciarNivel();
            delay(350);
          }
        }
      }

      // --- Render ---
      char hud[21];
      snprintf(hud, sizeof(hud), "L%1d LV%02d S%05d", vidas, nivel, score);
      lcd.setCursor(0, 0);
      lcd.print("                    ");
      lcd.setCursor(0, 0);
      lcd.print(hud);
      if (frutaActiva) {
        lcd.setCursor(18, 0);
        lcd.write(6);
      } else {
        lcd.setCursor(18, 0);
        lcd.write(5);
      }

      bool bocaAbierta = ((millis() / 120) % 2 == 0);
      bool parpadeoMiedo = ((millis() / 180) % 2 == 0);
      bool miedoAcabando = (miedoHasta > 0 && miedoHasta - millis() < 1400);

      for (int y = 0; y < H; y++) {
        lcd.setCursor(0, y + 1);
        for (int x = 0; x < W; x++) {
          bool dibujado = false;

          // Fantasmas
          for (int g = 0; g < G; g++) {
            if (gx[g] == x && gy[g] == y) {
              if (gEstado[g] == 2) lcd.write(7);
              else if (gEstado[g] == 1) {
                if (miedoAcabando && parpadeoMiedo) lcd.write(3);
                else lcd.write(4);
              } else lcd.write(3);
              dibujado = true;
              break;
            }
          }

          // Pac-Man
          if (!dibujado && pacX == x && pacY == y) {
            if (!bocaAbierta || dirPac == 0 || dirPac == 2) lcd.write(2);
            else if (dirPac == 1) lcd.write(0);
            else lcd.write(1);
            dibujado = true;
          }

          // Fruta
          if (!dibujado && frutaActiva && x == frutaX && y == frutaY) {
            lcd.write(6);
            dibujado = true;
          }

          if (!dibujado) {
            if (mapa[y][x] == 3) lcd.print('#');
            else if (mapa[y][x] == 2) lcd.print('o');
            else if (mapa[y][x] == 1) lcd.print('.');
            else lcd.print(' ');
          }
        }
      }

      delay(18);
    }

    if (salidaForzada) {
      break;
    }

    lcd.clear();
    lcd.setCursor(5, 0); lcd.print("PACMAN OVER");
    char b1[21];
    char b2[21];
    snprintf(b1, sizeof(b1), " SCORE: %05d       ", score);
    snprintf(b2, sizeof(b2), " LEVEL: %02d        ", nivel);
    lcd.setCursor(0, 1); lcd.print(b1);
    lcd.setCursor(0, 2); lcd.print(b2);
    lcd.setCursor(0, 3); lcd.print("START Exit MENU Retry");

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
  transicionBarrido();
  lcd.clear();
}


