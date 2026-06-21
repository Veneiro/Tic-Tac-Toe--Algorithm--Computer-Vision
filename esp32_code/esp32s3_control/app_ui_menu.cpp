#include "app_contracts.h"

/** @brief Bucle del menú principal: muestra un menú animado con selector de modo y efecto de barrido "PRESS START".
 *  Lee el interruptor de modo (AUTO/MANUAL), detecta combinaciones de easter-egg (MENU+JOY = Pacman, MENU = Snake)
 *  y bloquea hasta que se pulsa START. Llama a confirmarInicio() para los inicios en modo normal. */
void esperarSeleccionMenu()
{
  bool ultimoEstadoSwitch = !digitalRead(pinSwitch); 
  char ultimaFila3[21] = ""; // Guarda la línea entera para evitar parpadeos

  lcd.clear();

  while (digitalRead(pinStart) == LOW)
  {
    server.handleClient(); 
    unsigned long t = millis();

    // 1. ANIMACIÓN DE TÍTULO (Engranajes girando)
    bool frameEngranaje = (t / 250) % 2 == 0; 
    lcd.setCursor(3, 0);
    lcd.write(frameEngranaje ? 5 : 6); 
    lcd.print(" MAIN  MENU ");
    lcd.write(frameEngranaje ? 6 : 5); 

    // 2. DETECCIÓN DEL MODO
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

    // 3. ANIMACIÓN DE FLECHAS Y EFECTO "BARRIDO" (WIPE)
    char filaActual3[21] = "                    "; // 20 espacios en blanco
    
    // Flechas moviéndose (Adentro / Afuera)
    bool flechasDentro = (t / 400) % 2 == 0;
    if (flechasDentro) {
      filaActual3[2] = '>'; filaActual3[3] = '>';
      filaActual3[15] = '<'; filaActual3[16] = '<';
    } else {
      filaActual3[1] = '>'; filaActual3[2] = '>';
      filaActual3[16] = '<'; filaActual3[17] = '<';
    }

    // Lógica del Barrido de "PULSE START"
    const char* txt = "PRESS START";
    int longitudTexto = 11;
    int cicloText = t % 3000; // Ciclo total de 3 segundos
    
    int startIdx = 0; // Por dónde empieza a verse el texto
    int endIdx = 0;   // Por dónde termina de verse

    // Fase 1 (Aparece de Izquierda a Derecha)
    if (cicloText <= 1200) {
      startIdx = 0;
      endIdx = (cicloText * longitudTexto) / 1200;
    } 
    // Fase 2 (Se mantiene el texto completo visible)
    else if (cicloText <= 1800) {
      startIdx = 0;
      endIdx = longitudTexto;
    } 
    // Fase 3 (Desaparece de Izquierda a Derecha - Efecto Escoba)
    else {
      startIdx = ((cicloText - 1800) * longitudTexto) / 1200;
      endIdx = longitudTexto;
    }
    
    // Pegamos las letras en la posición central (columna 4)
    // Solo si están dentro del rango visible [startIdx, endIdx)
    for (int i = 0; i < longitudTexto; i++) {
      if (i >= startIdx && i < endIdx) {
        filaActual3[4 + i] = txt[i]; 
      }
    }

    // Imprimimos la fila entera SOLO SI HA CAMBIADO (Sin parpadeos)
    if (strcmp(filaActual3, ultimaFila3) != 0) {
      lcd.setCursor(0, 3);
      lcd.print(filaActual3);
      strcpy(ultimaFila3, filaActual3); 
    }

    delay(20); 
  }
  
  // --- CHEQUEO DE EASTER EGGS ---
  // MENU + JOY -> Pacman
  // MENU solo -> Snake
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

/** @brief Reproduce una animación de confirmación de modo en el LCD antes de entrar al juego.
 *  Para modo AUTO: muestra la secuencia de activación del robot con animación de parpadeo de ojos.
 *  Para modo MANUAL: muestra barras de calibración de ejes con simulación sinusoidal del joystick. */
void confirmarInicio()
{
  lcd.clear();

  if (modoAutomatico) 
  {
    // Fase 1: Apagado
    lcd.setCursor(0, 0); lcd.print("     ___            ");
    lcd.setCursor(0, 1); lcd.print("    [off]           ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|" ); lcd.write(4); lcd.print("          ");
    lcd.setCursor(0, 3); lcd.print("   ==| |==          ");
    delay(500);
    
    // Fase 2: Antenas arriba y encendiendo
    lcd.setCursor(0, 0); lcd.print("     _|_            ");
    lcd.setCursor(0, 1); lcd.print("    [-_-]           ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4);; lcd.print("          ");
    delay(400);

    // Fase 3: Totalmente encendido con texto
    lcd.setCursor(0, 0); lcd.print("     " ); lcd.write(4); lcd.print("|/            ");
    lcd.setCursor(0, 1); lcd.print("    [O_O]   AUTO    ");
    lcd.setCursor(0, 2); lcd.print("   /|:::|"); lcd.write(4);lcd.print("  MODE    ");
    delay(400);
    
    // Fase 4: Parpadeo de los ojos
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
    
    // Dibujamos el "chasis" de las barras de calibración
    lcd.setCursor(0, 1); lcd.print("X:[             ]");
    lcd.setCursor(0, 2); lcd.print("Y:[             ]");
    lcd.setCursor(1, 3); lcd.print(" Calibrating Axes ");

    // Simulamos que giran el joystick en un círculo 360º
    // Usamos seno y coseno para que se mueva suave de un lado a otro
    for (float t = 0; t <= 6.28; t += 0.35) 
    {
      // Calculamos la posición del cursor (centro en la col 9, amplitud de 6)
      int posX = 9 + 6 * sin(t); 
      int posY = 9 + 6 * cos(t);

      // Dibujamos el movimiento en la barra X
      lcd.setCursor(3, 1);
      for(int i = 3; i <= 15; i++) {
        if (i == posX) lcd.write(255); // El carácter 255 es un bloque cuadrado sólido (█)
        else lcd.print("-");
      }

      // Dibujamos el movimiento en la barra Y
      lcd.setCursor(3, 2);
      for(int i = 3; i <= 15; i++) {
        if (i == posY) lcd.write(255); 
        else lcd.print("-");
      }

      delay(60); 
    }

    // Mensaje final de confirmación
    lcd.setCursor(1, 3); 
    lcd.print("   SYSTEM READY!   ");
    delay(600);
  }
}


