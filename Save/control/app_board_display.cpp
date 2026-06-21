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

void actualizarLCD()
{
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
        bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
        if (esNueva) lcd.print(".");
        else if (tableroAnterior[i][j] == 0) lcd.print(" ");
        else if (tableroAnterior[i][j] == 1) lcd.print("X");
        else if (tableroAnterior[i][j] == 2) lcd.print("O");
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
        bool esNueva = (tablero[i][j] != tableroAnterior[i][j] && tablero[i][j] != 0);
        if (esNueva) lcd.write(3); // Carácter personalizado de destello
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

  // 3. Dibujo final y estático del tablero
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

  // --- LÓGICA DE LA COPA (SOLO SI HAY GANADOR REAL 1 o 2) ---
  int ganador = comprobarGanador(); // <--- CORREGIDO EL NOMBRE AQUÍ
  
  if (ganador == 1 || ganador == 2) {
    delay(1200);        // Pausa para ver la jugada final
    mostrarCopaASCII();   // Dibujamos la copa encima del tablero
    delay(1500);        // Tiempo para disfrutar el trofeo
  } 
  else if (ganador == 3) {
    // Si es empate, quizás solo una pausa corta antes de los fuegos
    delay(1500);
  }
}

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

// =========================================================
// FUNCIONES DEL SERVIDOR 
// =========================================================


