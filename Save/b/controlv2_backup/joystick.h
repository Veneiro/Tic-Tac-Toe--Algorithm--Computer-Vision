#ifndef JOYSTICK_H
#define JOYSTICK_H

#include <Arduino.h>

class Joystick
{
private:
    int pinX;
    int pinY;
    int pinButton;

    int valorX;
    int valorY;
    int valorZ;

    int xMin;
    int xCenter;
    int xMax;

    int yMin;
    int yCenter;
    int yMax;

    int deadZone;

    bool modoZ;

    bool lastButtonReading;
    bool buttonState;
    unsigned long lastDebounceTime;
    unsigned long debounceDelay;

    float mapToVelocity(int value,
                        int center,
                        int minVal,
                        int maxVal,
                        float Vmax,
                        bool invertDirection);

    void updateButtonToggle();

public:
    Joystick(int pinJoyX,
             int pinJoyY,
             int pinJoyButton,
             int joyXMin,
             int joyXCenter,
             int joyXMax,
             int joyYMin,
             int joyYCenter,
             int joyYMax);

    void begin();
    void update();

    int getValorX() const;
    int getValorY() const;
    int getValorZ() const;

    float getVx(float Vmax);
    float getVy(float Vmax);
    float getVz(float Vmax);

    bool isModoZ() const;

    void setDeadZone(int dz);
    void setDebounceDelay(unsigned long delayMs);
};

#endif
