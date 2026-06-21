#include "app_contracts.h"

// ─────────────────────────────────────────────────────────
// Helpers para la pantalla de verificación de tablero
// ─────────────────────────────────────────────────────────

/** @brief Muestra en el LCD el mensaje de estado adecuado para la verificación del tablero según las banderas actuales.
 *  Muestra mensajes distintos para: mano detectada + tablero sucio, solo mano,
 *  solo tablero sucio, o un mensaje genérico de "comprobando" mientras se espera el primer resultado. */
static void _lcdVerificacionMensaje()
{
  LCD_LOCK();
  lcd.setCursor(0, 0);
  lcd.print("== BOARD  CHECK  ===");

  if (verificarManoEnTablero && !verificarTableroLimpio)
  {
    lcd.setCursor(0, 1); lcd.print("  RETIRE LA MANO    ");
    lcd.setCursor(0, 2); lcd.print("  y limpie el       ");
    lcd.setCursor(0, 3); lcd.print("  TABLERO           ");
  }
  else if (verificarManoEnTablero)
  {
    lcd.setCursor(0, 1); lcd.print("  RETIRE LA MANO    ");
    lcd.setCursor(0, 2); lcd.print("  antes de empezar  ");
    lcd.setCursor(0, 3); lcd.print("                    ");
  }
  else if (!verificarTableroLimpio)
  {
    lcd.setCursor(0, 1); lcd.print("  LIMPIE EL         ");
    lcd.setCursor(0, 2); lcd.print("  TABLERO antes     ");
    lcd.setCursor(0, 3); lcd.print("  de comenzar       ");
  }
  else
  {
    // Aún no hemos recibido ningún resultado: mensaje genérico de espera
    lcd.setCursor(0, 1); lcd.print("  Comprobando...    ");
    lcd.setCursor(0, 2); lcd.print("                    ");
    lcd.setCursor(0, 3); lcd.print("  Por favor espere  ");
  }
  LCD_UNLOCK();
}

/** @brief Bucle bloqueante: envía peticiones /verificar repetidamente hasta que el tablero esté limpio
 *  y no se detecte ninguna mano. Muestra el estado de verificación en el LCD y reintenta en caso de tiempo de espera agotado o error de red. */
void esperarTableroLimpio()
{
  const unsigned long TIMEOUT_MS  = 9000;  // tiempo máx. esperando respuesta de cámara
  const unsigned long PAUSA_MS    =  800;  // pausa tras resultado negativo antes de reintentar

  verificarResultadoPendiente = false;
  verificarListo              = false;
  verificarTableroLimpio      = false;
  verificarManoEnTablero      = false;

  // Pantalla inicial de espera
  LCD_LOCK();
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("== BOARD  CHECK  ===");
  lcd.setCursor(0, 1); lcd.print("  Comprobando...    ");
  lcd.setCursor(0, 2); lcd.print("                    ");
  lcd.setCursor(0, 3); lcd.print("  Por favor espere  ");
  LCD_UNLOCK();

  bool esperandoRespuesta = false;
  unsigned long tiempoEnvio = 0;

  while (true)
  {
    server.handleClient();

    // Llegó respuesta de la cámara
    if (esperandoRespuesta && !verificarResultadoPendiente)
    {
      esperandoRespuesta = false;
      if (verificarListo)
        break;  // ¡Tablero listo, salimos!

      // No listo: actualizar mensaje y pausar antes de reintentar
      _lcdVerificacionMensaje();
      unsigned long finPausa = millis() + PAUSA_MS;
      while (millis() < finPausa) { server.handleClient(); delay(20); }
    }

    // Timeout de espera
    if (esperandoRespuesta && millis() - tiempoEnvio > TIMEOUT_MS)
    {
      Serial.println("[VERIFICAR] Timeout esperando respuesta de camara, reintentando...");
      esperandoRespuesta = false;
      verificarResultadoPendiente = false;
    }

    // Enviar nueva solicitud de verificación
    if (!esperandoRespuesta)
    {
      verificarResultadoPendiente = true;
      verificarListo              = false;
      tiempoEnvio                 = millis();

      if (solicitarVerificacion())
      {
        esperandoRespuesta = true;
      }
      else
      {
        // Error de red: breve pausa y reintento
        verificarResultadoPendiente = false;
        Serial.println("[VERIFICAR] Error de red, reintentando en 2s...");
        unsigned long espera = millis() + 2000;
        while (millis() < espera) { server.handleClient(); delay(20); }
      }
    }

    delay(20);
  }

  // Confirmar en pantalla antes de continuar con el flujo normal
  LCD_LOCK();
  lcd.clear();
  lcd.setCursor(0, 1); lcd.print("  Tablero listo!    ");
  lcd.setCursor(0, 2); lcd.print("  Iniciando partida ");
  LCD_UNLOCK();
  delay(1000);
}

/** @brief Bucle principal de juego: orquesta el modo automático (visión + robot) y el modo manual (joystick).
 *  En modo automático, gestiona la alternancia de turnos, las solicitudes de captura de cámara, el procesamiento del tablero,
 *  la escalada de música de jefe final y el menú de pausa. En modo manual, mapea la entrada del joystick
 *  a la velocidad del efector final y gestiona la activación de la pinza mediante el botón START. */
void ejecutarPartida1()
{
  vaciarTablero();
  robotServiceResetPieces();
  tableroPendiente = false;
  setSemaforo(false, false, false);

  int ganador = comprobarGanador();
  bool abortarPartida = false;        
  static bool ultimoEstadoMenu = HIGH; 

  if (modoAutomatico == true)
  {
    // Verificar que el tablero esté vacío y sin manos antes de empezar
    esperarTableroLimpio();

    // Sorteo de inicio: Jugador 1 (humano) o Jugador 2 (brazo robotico).
    turnoMaquina = (random(100) < 50);
    mostrarPantallaTurnoInicial(turnoMaquina);

    if (turnoMaquina) setSemaforo(true, false, false);
    else setSemaforo(false, false, true);

    // --- NUEVO: ANIMACIÓN TIPO ESCÁNER AL ENTRAR AL TABLERO ---
    animarEntradaTablero();
    juegoEnCurso = true;  // La tarea LCD empieza aquí, tras la animación de entrada
    actualizarLCD();

    bool ultimoEstadoStartAuto = digitalRead(pinStart);
    bool turnoRobotPendiente = turnoMaquina;
    bool capturaEnviada = false;
    int intentosCaptura = 0;
    unsigned long ultimoIntentoCaptura = 0;
    const unsigned long retardoReintentoCapturaMs = 1500;
    bossBattleActivo = false;

    while (ganador == 0 && !abortarPartida)
    {
      server.handleClient();

      // Arrancar boss battle desde el loop principal (no desde la tarea LCD)
      // para evitar llamar buzzerPlay() mientras se sostiene lcdMutex.
      if (!bossBattleActivo)
      {
        int libres = 0;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            if (tablero[i][j] == 0) libres++;
        if (libres <= 3 && libres > 0)
        {
          bossBattleActivo = true;
          buzzerPlay(CANCION_BOSS_BATTLE, true);
        }
      }

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
        esperarLiberacionBoton(pinMenu);
        // Bloquear la tarea de animación durante toda la interacción con el menú
        LCD_LOCK();
        abortarPartida = abrirMenuPausa();

        // --- LÓGICA DE SALIDA DRAMÁTICA (AUTO) ---
        if (abortarPartida)
        {
          setSemaforo(false, false, false);
          // El LCD_LOCK ya está activo (recursivo), no hace falta uno nuevo

          // Animación: Robot cayéndose y desactivándose
          lcd.clear();
          // Fotograma 1: De pie, cansado
          lcd.setCursor(5, 1); lcd.print("[>_<]");
          lcd.setCursor(3, 2); lcd.print("/|:::|"); lcd.write(4);
          delay(400);

          // Fotograma 2: Empezando a caer
          lcd.clear();
          lcd.setCursor(5, 2); lcd.print("[>_<]");
          lcd.setCursor(3, 3); lcd.write(4); lcd.print("|:::|"); lcd.print("/");
          delay(300);

          // Fotograma 3: En el suelo
          lcd.clear();
          lcd.setCursor(0, 3);
          lcd.print("   __(x_x)__/-*puff*");
          delay(1000);

          lcd.setCursor(0, 1); lcd.print(" AUTOMATIC MODE");
          lcd.setCursor(0, 2); lcd.print("   DEACTIVATED");
          delay(1000);
          LCD_UNLOCK();
          break;
        }

        // --- CORRECCIÓN AL VOLVER DE LA PAUSA ---
        actualizarLCD();  // actualizarLCD hace su propio LCD_LOCK recursivo interno
        LCD_UNLOCK();
      }
      ultimoEstadoMenu = lecturaMenu;

      if (!turnoMaquina && !abortarPartida && ganador == 0)
      {
        bool estadoStartAuto = digitalRead(pinStart);
        if (estadoStartAuto == HIGH && ultimoEstadoStartAuto == LOW)
        {
          esperarLiberacionBoton(pinStart);
          turnoMaquina = true;
          turnoRobotPendiente = true;
          setSemaforo(true, false, false);
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
    // Barrera de sincronización: esperar a que el frame de animación en curso
    // termine de escribir en la LCD antes de que manejarFinDeJuego tome el control.
    LCD_LOCK();
    LCD_UNLOCK();
    manejarFinDeJuego(ganador);
  }
  else
  {
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
    bool gripperClosed = false;
    bool lastStartState = HIGH;
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
          lcd.setCursor(3, 0); lcd.print("RECALIBRATING!!");
          lcd.setCursor(0, 1); lcd.print("X:[XXXXXXXXXXXXX]");
          lcd.setCursor(0, 2); lcd.print("Y:[XXXXXXXXXXXXX]");
          lcd.setCursor(1, 3); lcd.print("LEAVING MANUAL MODE");
          
          resetPosition();

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

      // --- 2.5. Botón START: alterna apertura/cierre de la pinza ---
      bool startState = digitalRead(pinStart);
      if (lastStartState == HIGH && startState == LOW)
      {
        gripperClosed = !gripperClosed;

        if (gripperClosed)
        {
          closeGripperSmooth();
        }
        else
        {
          openGripperSmooth();
        }
      }
      lastStartState = startState;

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

      // --- 3. Lectura del Joystick (valores via clase Joystick) ---
      joystick.update();

      float vx = joystick.getVx(V_max);
      float vy = joystick.getVy(V_max);
      float vz = joystick.getVz(V_max);

      interrupt_flag = false;
      while (interrupt_flag == false)
      {
      }

      // Mostramos la posición real del efector en la LCD
      float posX = my_robot.p.x;
      float posY = my_robot.p.y;
      float posZ = my_robot.p.z;

      // --- 4. Actualizar LCD según el modo ---
      if (!controlandoZ)
      {
        // === MODO X / Y ===
        int valorX = (int)(posX * 10.0f);
        int valorY = (int)(posY * 10.0f);

        // Aumentamos el umbral para evitar que el redondeo haga parpadear la LCD
        if (forzarDibujado || abs(valorX - ultimoX) > 1 || abs(valorY - ultimoY) > 1) {
            char bufferXY[21];
            snprintf(bufferXY, sizeof(bufferXY), " X:%5.1f Y:%5.1f ", posX, posY); 
            lcd.setCursor(0, 1);
            lcd.print(bufferXY);
            
            ultimoX = valorX;
            ultimoY = valorY;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Pos:  [ X / Y ]    ");
      }
      else
      {
        // === MODO Z ===
        int valorZ = (int)(posZ * 10.0f);
        if (forzarDibujado || abs(valorZ - ultimoZ) > 1) {
            char bufferZ[21];
            snprintf(bufferZ, sizeof(bufferZ), " Z:%5.1f           ", posZ); 
            lcd.setCursor(0, 1);
            lcd.print(bufferZ);
            
            ultimoZ = valorZ;
            forzarDibujado = false;
        }
        lcd.setCursor(0, 2);
        lcd.print(" Pos:  [   Z   ]    ");
      }

        // --- Movimiento del brazo usando velocidades del joystick ---
        LinearPosition candidate = target_position;
        candidate.x += vx * Ts;
        candidate.y += vy * Ts;
        candidate.z += vz * Ts;

        if (isInside(workspace, candidate))
        {
          IKResult my_solution = inverseKinematics(candidate);
          if (my_solution.hasSolution)
          {
            target_position = candidate;
            target_angle = my_solution.q;
            ok = true;
          }
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


