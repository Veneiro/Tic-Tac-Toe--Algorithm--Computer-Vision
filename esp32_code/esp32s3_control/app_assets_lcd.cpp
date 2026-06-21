#include "app_contracts.h"

byte charAparicion[8] = {
  0b00000, 0b00100, 0b01010, 0b10101, 0b01010, 0b00100, 0b00000, 0b00000
};

byte charRobot[8] = {
  0b00000, 0b01010, 0b11111, 0b10101, 0b11111, 0b10001, 0b01110, 0b00000
};

byte charTrofeo[8] = {
  0b11111, 0b10101, 0b10101, 0b01110, 0b00100, 0b00100, 0b01110, 0b00000
};

byte charJoy[8] = {
  0b00100, 0b01110, 0b00100, 0b00100, 0b01110, 0b11111, 0b11111, 0b00000
};

byte charEngranaje1[8] = {0b00000, 0b01010, 0b00100, 0b11111, 0b00100, 0b01010, 0b00000, 0b00000};
byte charEngranaje2[8] = {0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000, 0b00000};

byte charBarraInvertida[8] = {
  0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000, 0b00000, 0b00000
};

byte charSnakeHead[8] = {0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000};

// ── Caracteres de fuego (slots 5-7 durante animación de fin de partida) ─────
// charFireLight: chispa/brasa pequeña (slot 5)
byte charFireLight[8] = {
  0b00100,   //  . . * . .   punta
  0b01010,   //  . * . * .   bifurcación
  0b00100,   //  . . * . .   cuello
  0b00000,
  0b00000,
  0b00000,
  0b00000,
  0b00000
};

// charFireMed: llama media (slot 6)
byte charFireMed[8] = {
  0b00100,   //  . . * . .   punta
  0b01110,   //  . * * * .   ensanchando
  0b11011,   //  * * . * *   cuerpo con hueco
  0b01110,   //  . * * * .   base
  0b00000,
  0b00000,
  0b00000,
  0b00000
};

// charFireHot: llama intensa / base (slot 7)
byte charFireHot[8] = {
  0b01010,   //  . * . * .   puntas parpadeantes
  0b11111,   //  * * * * *   cuerpo lleno
  0b10101,   //  * . * . *   textura ondulante
  0b11111,   //  * * * * *   cuerpo
  0b11111,   //  * * * * *   cuerpo
  0b11111,   //  * * * * *   base
  0b01010,   //  . * . * .   brasas abajo
  0b00000
};

// ── Cara humana ──────────────────────────────────────────────────────────────
// Cara humana con ojos abiertos (slot 5 durante turno humano)
byte charHumanFace[8]  = {
  0b00000,   // vacío
  0b01110,   // cabeza arriba
  0b10001,   // frente
  0b11011,   // ojos:  ** **
  0b10001,   // mejillas
  0b10101,   // boca con comisuras
  0b01110,   // barbilla (forma sonrisa con fila anterior)
  0b00100    // cuello
};

// Cara humana con ojos cerrados / parpadeando (slot 6 durante turno humano)
byte charHumanBlink[8] = {
  0b00000,
  0b01110,
  0b10001,
  0b11111,   // ojos cerrados (línea llena)
  0b10001,
  0b10101,
  0b01110,
  0b00100
};
