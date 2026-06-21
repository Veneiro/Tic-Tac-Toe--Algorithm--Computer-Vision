#ifndef GRIPPER_H
#define GRIPPER_H

#include <Arduino.h>

void beginGripper();

bool enableGripperServo();
void disableGripperServo();
bool isGripperEnabled();

void openGripper();
void closeGripper();

void openGripperSmooth();
void closeGripperSmooth();

void closeGripperAndRelease();
void openGripperAndRelease();
void openGripperFromReleased();

void moveGripperPulse(int pulseUs);
void moveGripperSmoothPulse(int targetPulseUs);

int getGripperPulseUs();

#endif