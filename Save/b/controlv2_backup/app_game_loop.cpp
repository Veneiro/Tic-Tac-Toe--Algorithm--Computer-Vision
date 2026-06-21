#include "app_contracts.h"
#include "joystick.h"

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
				esperarLiberacionBoton(pinMenu);
				abortarPartida = abrirMenuPausa();

				if (abortarPartida)
				{
					setSemaforo(false, false, false);

					lcd.clear();
					lcd.setCursor(5, 1); lcd.print("[>_<]");
					lcd.setCursor(3, 2); lcd.print("/|:::|"); lcd.write(4);
					delay(400);

					lcd.clear();
					lcd.setCursor(5, 2); lcd.print("[>_<]");
					lcd.setCursor(3, 3); lcd.write(4); lcd.print("|:::|"); lcd.print("/");
					delay(300);

					lcd.clear();
					lcd.setCursor(0, 3);
					lcd.print("   __(x_x)__/-*puff*");
					delay(1000);

					lcd.setCursor(0, 1); lcd.print(" AUTOMATIC MODE");
					lcd.setCursor(0, 2); lcd.print("   DEACTIVATED");
					delay(1000);
					break;
				}

				if (!abortarPartida)
				{
					lcd.clear();
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

		if (abortarPartida)
		{
			return;
		}
		manejarFinDeJuego(ganador);
	}
	else
	{
		// ======================================================
		// MODO MANUAL CON JOYSTICK
		// ======================================================
		setSemaforo(false, true, false);

		lcd.clear();
		lcd.setCursor(2, 0);
		lcd.print("-- MANUAL MODE --");
		lcd.setCursor(0, 3);
		lcd.print(" [MENU] for pause ");

		bool forzarDibujado = true;
		bool ultimoModoZ = joystick.isModoZ();
		float ultimoTx = 9999.0f;
		float ultimoTy = 9999.0f;
		float ultimoTz = 9999.0f;
		bool gripperClosed = false;
		bool lastStartState = HIGH;

		while (!abortarPartida)
		{
			server.handleClient();

			bool lecturaMenu = digitalRead(pinMenu);
			if (lecturaMenu == HIGH && ultimoEstadoMenu == LOW)
			{
				esperarLiberacionBoton(pinMenu);
				abortarPartida = abrirMenuPausa();

				if (abortarPartida)
				{
					lcd.clear();
					lcd.setCursor(3, 0); lcd.print("UNSTABLE SYSTEM!");
					lcd.setCursor(0, 1); lcd.print("X:[-------------]");
					lcd.setCursor(0, 2); lcd.print("Y:[-------------]");

					for (int i = 0; i < 25; i++)
					{
						int randomX = random(3, 16);
						int randomY = random(3, 16);

						lcd.setCursor(3, 1); lcd.print("-------------");
						lcd.setCursor(3, 2); lcd.print("-------------");

						lcd.setCursor(randomX, 1); lcd.write(255);
						lcd.setCursor(randomY, 2); lcd.write(255);

						delay(60);
					}

					lcd.clear();
					lcd.setCursor(3, 0); lcd.print("CRITICAL ERROR!!");
					lcd.setCursor(0, 1); lcd.print("X:[XXXXXXXXXXXXX]");
					lcd.setCursor(0, 2); lcd.print("Y:[XXXXXXXXXXXXX]");
					lcd.setCursor(1, 3); lcd.print("LEAVING MANUAL MODE");

					delay(1500);
					break;
				}

				if (!abortarPartida)
				{
					lcd.clear();
					lcd.setCursor(2, 0);
					lcd.print("--- MANUAL MODE ---");
					lcd.setCursor(0, 3);
					lcd.print(" [MENU] for pause ");
					forzarDibujado = true;
				}
			}
			ultimoEstadoMenu = lecturaMenu;

			if (abortarPartida)
			{
				break;
			}

			joystick.update();

			float vx = joystick.getVx(V_max);
			float vy = joystick.getVy(V_max);
			float vz = joystick.getVz(V_max);

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

			LinearPosition candidate = target_position;
			candidate.x += vx * Ts;
			candidate.y += vy * Ts;
			candidate.z += vz * Ts;

			if (vx != 0.0f || vy != 0.0f || vz != 0.0f)
			{
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
			}

			bool modoZActual = joystick.isModoZ();
			if (modoZActual != ultimoModoZ)
			{
				ultimoModoZ = modoZActual;
				forzarDibujado = true;
			}

			if (forzarDibujado || fabsf(target_position.x - ultimoTx) > 0.05f || fabsf(target_position.y - ultimoTy) > 0.05f || fabsf(target_position.z - ultimoTz) > 0.05f)
			{
				char bufferLinea1[21];
				char bufferLinea2[21];

				if (!modoZActual)
				{
					snprintf(bufferLinea1, sizeof(bufferLinea1), " X:%6.2f Y:%6.2f", target_position.x, target_position.y);
					snprintf(bufferLinea2, sizeof(bufferLinea2), " AXIS: [ X / Y ]   ");
				}
				else
				{
					snprintf(bufferLinea1, sizeof(bufferLinea1), " Z:%6.2f          ", target_position.z);
					snprintf(bufferLinea2, sizeof(bufferLinea2), " AXIS: [   Z   ]   ");
				}

				lcd.setCursor(0, 1);
				lcd.print(bufferLinea1);
				lcd.setCursor(0, 2);
				lcd.print(bufferLinea2);

				ultimoTx = target_position.x;
				ultimoTy = target_position.y;
				ultimoTz = target_position.z;
				forzarDibujado = false;
			}

			delay(20);
		}

		juegoEnCurso = false;
		setSemaforo(false, false, false);

		if (abortarPartida)
		{
			return;
		}
	}
}
