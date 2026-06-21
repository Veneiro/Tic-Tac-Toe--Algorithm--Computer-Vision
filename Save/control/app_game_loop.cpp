#include "app_contracts.h"

void ejecutarPartida1()
{
  vaciarTablero();
  robotServiceResetPieces();
  tableroPendiente = false;
  juegoEnCurso = true;
  setSemaforo(false, false, false);

  int ganador = comprobarGanador();
  bool abortarPartida = false;        
  static bool ultimoEstadoMenu = HIGH; 

  if (modoAutomatico == true)
  {
    // Sorteo de inicio: Jugador 1 (humano) o Jugador 2 (brazo robotico).
    turnoMaquina = (random(100) < 50);
    mostrarPantallaTurnoInicial(turnoMaquina);

    if (turnoMaquina) setSemaforo(true, false, false);
    else setSemaforo(false, false, true);

    // --- NUEVO: ANIMACIÓN TIPO ESCÁNER AL ENTRAR AL TABLERO ---
    animarEntradaTablero(); 
    actualizarLCD();
    dibujarDecoracionTurnoAuto();

    bool ultimoEstadoStartAuto = digitalRead(pinStart);
    bool turnoRobotPendiente = turnoMaquina;
    bool capturaEnviada = false;
    int intentosCaptura = 0;
    unsigned long ultimoIntentoCaptura = 0;
    const unsigned long retardoReintentoCapturaMs = 1500;

    while (ganador == 0 && !abortarPartida)
    {
      server.handleClient(); 

      if (tableroPendiente && !abortarPartida)
      {
        String tableroLocal = tableroRecibidoHttp;
        tableroPendiente = false;

        procesarEntradaTablero(tableroLocal);
        ganador = comprobarGanador();

        if (ganador == 0)
        {
          turnoMaquina = false;
          turnoRobotPendiente = false;
          capturaEnviada = false;
          setSemaforo(false, false, true);
          dibujarDecoracionTurnoAuto();
          Serial.println("[TURN] Turno jugador: pulsa START para continuar");
        }
      }

      if (turnoMaquina && turnoRobotPendiente && !capturaEnviada && !abortarPartida)
      {
        if (millis() - ultimoIntentoCaptura >= retardoReintentoCapturaMs)
        {
          ultimoIntentoCaptura = millis();
          intentosCaptura++;

          if (solicitarCapturaCamara())
          {
            capturaEnviada = true;
            intentosCaptura = 0;
          }
          else
          {
            Serial.println("[CAM] Fallo al solicitar captura");
          }
        }
      }

      bool lecturaMenu = digitalRead(pinMenu);
      if (lecturaMenu == HIGH && ultimoEstadoMenu == LOW)
      {            
        esperarLiberacionBoton(pinMenu); // Esperar que suelte antes de entrar
        abortarPartida = abrirMenuPausa();
        
        // --- LÓGICA DE SALIDA DRAMÁTICA (AUTO) ---
        if (abortarPartida)
        {
          setSemaforo(false, false, false);

          // Animación: Robot cayéndose y desactivándose
          lcd.clear();
          // Fotograma 1: De pie, cansado
          lcd.setCursor(5, 1); lcd.print("[>_<]");
          lcd.setCursor(3, 2); lcd.print("/|:::|"); lcd.write(4);
          delay(400);

          // Fotograma 2: Empezando a caer (bajar una fila, cambiar brazos)
          lcd.clear();
          lcd.setCursor(5, 2); lcd.print("[>_<]");
          lcd.setCursor(3, 3); lcd.write(4); lcd.print("|:::|"); lcd.print("/");
          delay(300);

          // Fotograma 3: En el suelo, totalmente plano y desactivado
          lcd.clear();
          lcd.setCursor(0, 3); // Empezamos desde el borde izquierdo
          lcd.print("   __(x_x)__/-*puff*"); // Corregido con espacios perfectos
          delay(1000);
          
          lcd.setCursor(0, 1); lcd.print(" AUTOMATIC MODE");
          lcd.setCursor(0, 2); lcd.print("   DEACTIVATED");
          delay(1000);
          break; // Salir del while del juego
        }
        
        // --- CORRECCIÓN AL VOLVER DE LA PAUSA ---
        if (!abortarPartida)
        {
          lcd.clear();
          // Ya no ponemos "ESTADO DEL TABLERO", porque actualizarLCD()
          // ahora se encarga de poner su propio título decorado.
          actualizarLCD();
          dibujarDecoracionTurnoAuto();
        }
      }
      ultimoEstadoMenu = lecturaMenu;

      if (turnoMaquina)
      {
        dibujarDecoracionTurnoAuto();
      }

      if (!turnoMaquina && !abortarPartida && ganador == 0)
      {
        bool estadoStartAuto = digitalRead(pinStart);
        if (estadoStartAuto == HIGH && ultimoEstadoStartAuto == LOW)
        {
          esperarLiberacionBoton(pinStart);
          turnoMaquina = true;
          turnoRobotPendiente = true;
          setSemaforo(true, false, false);
          dibujarDecoracionTurnoAuto();
          Serial.println("[TURN] Turno maquina: esperando captura de la camara");
        }
        ultimoEstadoStartAuto = estadoStartAuto;
      }
    }

    juegoEnCurso = false; 
    setSemaforo(false, false, false);

    if (abortarPartida) {
      return; 
    }
    manejarFinDeJuego(ganador);
  }
  else
  {
    // ======================================================
    // MODO MANUAL (Con cambio a eje Z)
    // ======================================================
    setSemaforo(false, true, false);

    lcd.clear();
    lcd.setCursor(2, 0);
    lcd.print("-- MANUAL MODE --");
    lcd.setCursor(0, 3);
    lcd.print(" [MENU] for pause ");

    int ultimoX = -1;
    int ultimoY = -1;
    int ultimoZ = -1;
    
    bool controlandoZ = false; // Estado: False = XY, True = Z
    bool antBotonJoy = digitalRead(pinJoyButton);
    bool forzarDibujado = true; // Flag para obligar a repintar la LCD al cambiar de modo

    while (!abortarPartida)
    {
      server.handleClient(); 

      // --- 1. Detección de pausa ---
      bool lecturaMenu = digitalRead(pinMenu);
      if (lecturaMenu == HIGH && ultimoEstadoMenu == LOW)
      {            
        esperarLiberacionBoton(pinMenu); 
        abortarPartida = abrirMenuPausa();
        
        // --- LÓGICA DE SALIDA DRAMÁTICA (MANUAL) CORREGIDA ---
        if (abortarPartida)
        {
          lcd.clear();
          // Centramos el título (22 caracteres -> usamos 20: "! SISTEMA INESTABLE !")
          lcd.setCursor(3, 0); lcd.print("UNSTABLE SYSTEM!");
          
          // Dibujamos las barras base fijas (Columna 0 a 16)
          lcd.setCursor(0, 1); lcd.print("X:[-------------]");
          lcd.setCursor(0, 2); lcd.print("Y:[-------------]");

          // Bucle de agitación (25 frames)
          for (int i = 0; i < 25; i++) {
            // Rango seguro: de la columna 3 a la 15 (dentro de los corchetes)
            int randomX = random(3, 16); 
            int randomY = random(3, 16);

            // 1. Limpiamos solo el interior de las barras para evitar parpadeo total
            lcd.setCursor(3, 1); lcd.print("-------------");
            lcd.setCursor(3, 2); lcd.print("-------------");

            // 2. Dibujamos el bloque de error (ASCII 255) en la nueva posición aleatoria
            lcd.setCursor(randomX, 1); lcd.write(255); 
            lcd.setCursor(randomY, 2); lcd.write(255);
            
            delay(60); 
          }

          // --- ESTADO FINAL DE ERROR ---
          lcd.clear();
          lcd.setCursor(3, 0); lcd.print("CRITICAL ERROR!!");
          lcd.setCursor(0, 1); lcd.print("X:[XXXXXXXXXXXXX]");
          lcd.setCursor(0, 2); lcd.print("Y:[XXXXXXXXXXXXX]");
          lcd.setCursor(1, 3); lcd.print("LEAVING MANUAL MODE");
          
          delay(1500);
          break; // Salir del while del manual
        }
        
        if (!abortarPartida)
        {
          lcd.clear();
          lcd.setCursor(2, 0);
          lcd.print("--- MANUAL MODE ---");
          lcd.setCursor(0, 3);
          lcd.print(" [MENU] for pause ");
          forzarDibujado = true; // Forzar repintado tras volver de la pausa
        }
      }
      ultimoEstadoMenu = lecturaMenu;

      if (abortarPartida) break;

      // --- 2. Detección del botón del Joystick (Cambio de Modo) ---
      bool actBotonJoy = digitalRead(pinJoyButton);
      // Detectamos si lo acabas de pulsar (Flanco de bajada)
      if (actBotonJoy == LOW && antBotonJoy == HIGH) 
      {
        controlandoZ = !controlandoZ; // Alternar entre XY y Z
        forzarDibujado = true;        // Limpiar la línea de valores en la LCD
        delay(50);                    // Anti-rebote mecánico
      }
      antBotonJoy = actBotonJoy;

      // --- 3. Lectura del Joystick (Valores Crudos) ---
      int valorX = analogRead(pinJoyX);
      int valorY = analogRead(pinJoyY);
      
      // Como pediste, usamos directamente el valor crudo en Y para la Z
      int valorZ = valorY; 

      // --- 4. Actualizar LCD según el modo ---
      if (!controlandoZ)
      {
        // === MODO X / Y ===
        // Aumentamos el umbral (> 15) para evitar que el ruido del ADC haga parpadear la LCD
        if (forzarDibujado || abs(valorX - ultimoX) > 15 || abs(valorY - ultimoY) > 15) {
            char bufferXY[21];
            // Usamos %4d para reservar 4 cifras (0-4095) y quitamos el símbolo de %
            snprintf(bufferXY, sizeof(bufferXY), " X:%4d   Y:%4d   ", valorX, valorY); 
            lcd.setCursor(0, 1);
            lcd.print(bufferXY);
            
            ultimoX = valorX;
            ultimoY = valorY;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Axes: [ X / Y ]    ");
      }
      else
      {
        // === MODO Z ===
        if (forzarDibujado || abs(valorZ - ultimoZ) > 15) {
            char bufferZ[21];
            // Usamos %4d para la Z
            snprintf(bufferZ, sizeof(bufferZ), " Z:%4d            ", valorZ); 
            lcd.setCursor(0, 1);
            lcd.print(bufferZ);
            
            ultimoZ = valorZ;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Axis:  [   Z   ]    ");
      }

      delay(20); 
    }

    juegoEnCurso = false; 
  setSemaforo(false, false, false);

    if (abortarPartida) {
      return; 
    }
  }
}


