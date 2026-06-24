#include "app_contracts.h"

/** @brief Parsea una cadena con formato "{r,c,c;r,c,c;r,c,c}" en la matriz tablero[][].
 *  @param entrada Representación en cadena del tablero 3x3 usando dígitos del 0 al 2
 *                 separados por comas (columnas) y punto y coma (filas). */
void leetablero(String entrada)
{
  int inicio = entrada.indexOf('{');
  int fin = entrada.indexOf('}');

  if (inicio == -1 || fin == -1)
  {
    Serial.println("Error: Invalid string format");
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

/** @brief Muestra una animación de entrada en el LCD para la vista del tablero (efecto escáner/revelado).
 *  Muestra el título letra a letra y luego dibuja el marco del tablero fila a fila. */
void animarEntradaTablero()
{
  buzzerPlay(CANCION_RACE_FANFARE, false);
  lcd.clear();
  
  // Fase 1: El título aparece letra a letra (Efecto escáner terminal)
  // Usamos 20 caracteres exactos para centrarlo perfecto
  String titulo = "===  [ BOARD ]   ===";
  lcd.setCursor(0, 0);
  for(int i = 0; i < titulo.length(); i++) {
    lcd.print(titulo[i]);
    delay(30);
  }
  delay(150);

  // Fase 2: Los corchetes exteriores caen fila por fila (Perfectamente centrados)
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(0, i);
    lcd.print("      [     ]       "); 
    delay(120);
  }

  // Fase 3: Las barras separadoras se dibujan como un láser de arriba a abajo
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(8, i); lcd.print("|");
    lcd.setCursor(10, i); lcd.print("|");
    delay(120);
  }
  
  delay(250); // Pequeña pausa de cortesía antes de que la cámara empiece a leer
}

/** @brief Dibuja el arte ASCII del trofeo en las filas 1–3 del LCD y lo mantiene durante 2 segundos. */
void mostrarCopaASCII() {
  // Limpiamos las 3 filas del tablero antes de dibujar para evitar basura
  for (int i = 1; i <= 3; i++) {
    lcd.setCursor(0, i);
    lcd.print("                    ");
  }

  // FILA 1: /(-------)\
  // 5 espacios + / + (-------) + \ + 4 espacios = 20
  lcd.setCursor(0, 1);
  lcd.print("    /(------)");
  lcd.write(4);
  lcd.print("    ");

  // FILA 2: \_) # 1 (_/
  // 6 espacios + \ + _) # 1 (_/ + 3 espacios = 20
  lcd.setCursor(0, 2);
  lcd.print("    ");
  lcd.write(4);
  lcd.print("_) #1 (_/   ");

  // FILA 3: (_____)
  // 6 espacios + (_____) + 7 espacios = 20
  lcd.setCursor(0, 3);
  lcd.print("      (____)       ");

  delay(2000); // Pausa para ver la copa bien
}

/** @brief Actualiza el LCD con el estado actual del tablero y anima las fichas recién colocadas.
 *  Adquiere LCD_LOCK, ejecuta una animación de aparición en dos fases para las fichas nuevas
 *  y luego dibuja el estado estático final del tablero. */
void actualizarLCD()
{
  LCD_LOCK();
  // 1. Comprobamos si hay alguna ficha nueva que animar
  bool hayCambio = false;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0) {
        hayCambio = true;
      }
    }
  }

  // 2. Decoración superior elegante (20 caracteres exactos)
  lcd.setCursor(0, 0);
  lcd.print("===  [ BOARD ]   ===");

  // 3. ANIMACIÓN DE APARICIÓN (Solo se ejecuta si detecta una ficha nueva)
  if (hayCambio) {
    // FASE 1: Un pequeño puntito donde va a aparecer la ficha
    for (int i = 0; i < 3; i++) {
      lcd.setCursor(0, i + 1);
      lcd.print("      ["); // Espacios ajustados para centrar
      for (int j = 0; j < 3; j++) {
        bool esNueva = (tablero[j][2 - i] != tableroAnterior[j][2 - i] && tablero[j][2 - i] != 0);
        if (esNueva) lcd.print(".");
        else if (tableroAnterior[j][2 - i] == 0) lcd.print(" ");
        else if (tableroAnterior[j][2 - i] == 1) lcd.print("X");
        else if (tableroAnterior[j][2 - i] == 2) lcd.print("O");
        if (j < 2) lcd.print("|");
      }
      lcd.print("]       "); // Limpia la basura de la derecha
      if (modoAutomatico) dibujarDecoracionTurnoAuto();
    }
    delay(150); 

    // FASE 2: El destello de luz
    for (int i = 0; i < 3; i++) {
      lcd.setCursor(0, i + 1);
      lcd.print("      [");
      for (int j = 0; j < 3; j++) {
        bool esNueva = (tablero[j][2 - i] != tableroAnterior[j][2 - i] && tablero[j][2 - i] != 0);
        if (esNueva) lcd.write(3); // Carácter personalizado de destello
        else if (tableroAnterior[j][2 - i] == 0) lcd.print(" ");
        else if (tableroAnterior[j][2 - i] == 1) lcd.print("X");
        else if (tableroAnterior[j][2 - i] == 2) lcd.print("O");
        if (j < 2) lcd.print("|");
      }
      lcd.print("]       ");
      if (modoAutomatico) dibujarDecoracionTurnoAuto();
    }
    delay(150); 
  }

  // 3. Dibujo final y estático del tablero
  lcd.setCursor(0, 0);
  lcd.print("===  [ BOARD ]   ===");
  for (int i = 0; i < 3; i++) {
    lcd.setCursor(0, i + 1);
    lcd.print("      [");
    for (int j = 0; j < 3; j++) {
      if (tablero[j][2 - i] == 0) lcd.print(" ");
      else if (tablero[j][2 - i] == 1) lcd.print("X");
      else if (tablero[j][2 - i] == 2) lcd.print("O");
      if (j < 2) lcd.print("|");
      tableroAnterior[j][2 - i] = tablero[j][2 - i];
    }
    lcd.print("]       ");
    if (modoAutomatico) dibujarDecoracionTurnoAuto();
  }

  LCD_UNLOCK();
}

/** @brief Comprueba todas las filas, columnas y diagonales de tablero[][] en busca de un ganador.
 *  @return 1 si gana el jugador 1 (X), 2 si gana el jugador 2 (O), 3 en caso de empate, 0 si la partida continúa. */
int comprobarGanador()
{
  // 1. Comprobar Filas
  for (int i = 0; i < 3; i++)
  {
    if (tablero[i][0] != 0 && tablero[i][0] == tablero[i][1] && tablero[i][1] == tablero[i][2])
    {
      return tablero[i][0];
    }
  }

  // 2. Comprobar Columnas
  for (int i = 0; i < 3; i++)
  {
    if (tablero[0][i] != 0 && tablero[0][i] == tablero[1][i] && tablero[1][i] == tablero[2][i])
    {
      return tablero[0][i];
    }
  }

  // 3. Comprobar Diagonal Principal (\)
  if (tablero[0][0] != 0 && tablero[0][0] == tablero[1][1] && tablero[1][1] == tablero[2][2])
  {
    return tablero[0][0];
  }

  // 4. Comprobar Diagonal Inversa (/)
  if (tablero[0][2] != 0 && tablero[0][2] == tablero[1][1] && tablero[1][1] == tablero[2][0])
  {
    return tablero[0][2];
  }

  // --- NUEVA LÓGICA: Comprobar si está lleno ---
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
    return 3; // EMPATE
  }

  return 0; // El juego sigue
}

/** @brief Imprime el estado actual de tablero[][] en la consola Serial para depuración. */
void printBoardSerial()
{
  Serial.println("Parsed matrix:");
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

/** @brief Resetea tablero[][] y tableroAnterior[][] a todos ceros (tablero vacío). */
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


