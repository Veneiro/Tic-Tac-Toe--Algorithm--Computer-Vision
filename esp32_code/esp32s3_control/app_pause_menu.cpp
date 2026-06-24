#include "app_contracts.h"

/** @brief Muestra la pantalla de pausa con una animación de apertura y espera la decisión del jugador.
 *  START sale del juego; MENU reanuda la partida. Incluye una animación de parpadeo inactivo mientras espera.
 *  @return Verdadero si el jugador eligió abortar la partida (START pulsado); falso para reanudar (MENU pulsado). */
bool abrirMenuPausa()
{
  // --- 1. Animación de Apertura (Cortina desde el centro hacia afuera) ---
  for (int i = 0; i < 10; i++) {
    for (int r = 0; r < 4; r++) {
      lcd.setCursor(9 - i, r); lcd.write(255);
      lcd.setCursor(10 + i, r); lcd.write(255);
    }
    delay(15);
  }
  for (int i = 0; i < 10; i++) {
    for (int r = 0; r < 4; r++) {
      lcd.setCursor(i, r); lcd.print(" ");
      lcd.setCursor(19 - i, r); lcd.print(" ");
    }
    delay(15);
  }
  
  // --- 2. Textos base bien centrados ---
  lcd.setCursor(4, 0); lcd.print("-[ PAUSED ]-");
  lcd.setCursor(3, 2); lcd.print("START: Exit");
  lcd.setCursor(3, 3); lcd.print("MENU:  Resume");

  bool antStartPausa = digitalRead(pinStart);
  bool antMenuPausa = digitalRead(pinMenu);
  
  delay(200); // Pausa visual de cortesía
  unsigned long startPausa = millis();

  while (true)
  {
    server.handleClient(); 

    // --- 3. Animación en idle (Parpadeo arcade) ---
    unsigned long t = millis() - startPausa;
    bool frame = (t / 300) % 2 == 0; // Cambia cada 300ms
    
    // Animación del título superior
    lcd.setCursor(2, 0);
    if (frame) {
      lcd.print(">>-[ PAUSED ]-<<");
    } else {
      lcd.print("  -[ PAUSED ]-  ");
    }
    
    // Flechas indicadoras en las opciones
    lcd.setCursor(1, 2); lcd.print(frame ? ">" : " ");
    lcd.setCursor(1, 3); lcd.print(frame ? ">" : " ");

    // --- 4. Lectura de botones ---
    bool actStart = digitalRead(pinStart);
    bool actMenu = digitalRead(pinMenu);

    // FLANCO EN START -> SALIR DE LA PARTIDA
    if (actStart == HIGH && antStartPausa == LOW)
    {
      esperarLiberacionBoton(pinStart); 
      transicionBarrido(); // Animación de cierre
      return true; // ABORTAR
    }
    antStartPausa = actStart;

    // FLANCO EN MENU -> CONTINUAR PARTIDA
    if (actMenu == HIGH && antMenuPausa == LOW)
    {
      esperarLiberacionBoton(pinMenu); 
      transicionBarrido(); // Animación de cierre
      return false; // CONTINUAR
    }
    antMenuPausa = actMenu;
    
    delay(20);
  }
}

// Función auxiliar para no repetir los mensajes de ganador

