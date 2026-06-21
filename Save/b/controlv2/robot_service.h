#pragma once

#include <Arduino.h>

void robotServiceInit();
void robotServiceResetPieces();
bool robotServiceMoveToCell(int fila, int columna);
void robotServiceApplyBoardDelta(int previousBoard[3][3], int currentBoard[3][3]);
