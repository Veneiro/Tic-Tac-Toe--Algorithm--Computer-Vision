#include "joystick.h"

Joystick::Joystick(int pinJoyX,
                   int pinJoyY,
                   int pinJoyButton,
                   int joyXMin,
                   int joyXCenter,
                   int joyXMax,
                   int joyYMin,
                   int joyYCenter,
                   int joyYMax)
{
    pinX = pinJoyX;
    pinY = pinJoyY;
    pinButton = pinJoyButton;

    valorX = joyXCenter;
    valorY = joyYCenter;
    valorZ = joyYCenter;

    xMin = joyXMin;
    xCenter = joyXCenter;
    xMax = joyXMax;

    yMin = joyYMin;
    yCenter = joyYCenter;
    yMax = joyYMax;

    deadZone = 80;

    modoZ = false;

    lastButtonReading = HIGH;
    buttonState = HIGH;
    lastDebounceTime = 0;
    debounceDelay = 50;
}

void Joystick::begin()
{
    pinMode(pinX, INPUT);
    pinMode(pinY, INPUT);
    pinMode(pinButton, INPUT_PULLUP);

    analogReadResolution(12);
}

void Joystick::update()
{
    updateButtonToggle();

    int lecturaX = analogRead(pinX);
    int lecturaY = analogRead(pinY);

    if (modoZ)
    {
        valorX = xCenter;
        valorY = yCenter;
        valorZ = lecturaY;
    }
    else
    {
        valorX = lecturaX;
        valorY = lecturaY;
        valorZ = yCenter;
    }
}

void Joystick::updateButtonToggle()
{
    bool reading = digitalRead(pinButton);

    if (reading != lastButtonReading)
    {
        lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay)
    {
        if (reading != buttonState)
        {
            buttonState = reading;

            if (buttonState == LOW)
            {
                modoZ = !modoZ;
            }
        }
    }

    lastButtonReading = reading;
}

float Joystick::mapToVelocity(int value,
                              int center,
                              int minVal,
                              int maxVal,
                              float Vmax,
                              bool invertDirection)
{
    int delta = value - center;

    if (abs(delta) < deadZone)
    {
        return 0.0f;
    }

    float v;

    if (delta > 0)
    {
        v = ((float)(value - center) / (float)(maxVal - center)) * Vmax;
    }
    else
    {
        v = ((float)(value - center) / (float)(center - minVal)) * Vmax;
    }

    if (invertDirection)
    {
        v = -v;
    }

    v = constrain(v, -Vmax, Vmax);

    return v;
}

int Joystick::getValorX() const
{
    return valorX;
}

int Joystick::getValorY() const
{
    return valorY;
}

int Joystick::getValorZ() const
{
    return valorZ;
}

float Joystick::getVx(float Vmax)
{
    return mapToVelocity(valorX, xCenter, xMin, xMax, Vmax, false);
}

float Joystick::getVy(float Vmax)
{
    return mapToVelocity(valorY, yCenter, yMin, yMax, Vmax, false);
}

float Joystick::getVz(float Vmax)
{
    return mapToVelocity(valorZ, yCenter, yMin, yMax, Vmax, true);
}

bool Joystick::isModoZ() const
{
    return modoZ;
}

void Joystick::setDeadZone(int dz)
{
    deadZone = dz;
}

void Joystick::setDebounceDelay(unsigned long delayMs)
{
    debounceDelay = delayMs;
}
