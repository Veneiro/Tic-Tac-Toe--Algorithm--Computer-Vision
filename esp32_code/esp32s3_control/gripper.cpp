#include <Arduino.h>
#include "config.h"
#include "gripper.h"

// ===================== CONFIGURACIÓN DE PINZA =====================

const int GRIPPER_OPEN_US  = 1200;
const int GRIPPER_CLOSE_US = 1905;

const int SERVO_FREQ = 50;
const int SERVO_RESOLUTION = 14;

const int GRIPPER_HOLD_CLOSE_MS = 400;
const int GRIPPER_HOLD_OPEN_MS  = 300;

static int gripperPulseUs = GRIPPER_OPEN_US;
static bool gripperEnabled     = false;
static bool gripperInitialized = false;

// ===================== FUNCIONES INTERNAS =====================

static uint32_t pulseUsToDuty(int pulseUs)
{
    pulseUs = constrain(pulseUs, 500, 2500);

    uint32_t maxDuty = (1UL << SERVO_RESOLUTION) - 1;

    // 50 Hz -> periodo = 20 000 us
    return (uint32_t)((pulseUs / 20000.0f) * maxDuty);
}

static void forceGripperPinLow()
{
    pinMode(PIN_GRIPPER, OUTPUT);
    digitalWrite(PIN_GRIPPER, LOW);
}

// ===================== CONTROL DE ACTIVACIÓN =====================

bool enableGripperServo()
{
    if (gripperEnabled) return true;

    if (!gripperInitialized)
    {
        // Adjuntar LEDC una sola vez con canal explícito (2) para no
        // interferir con los buzzers que usan canales 0 y 1.
        forceGripperPinLow();
        delay(5);

        bool ok = ledcAttachChannel(PIN_GRIPPER, SERVO_FREQ, SERVO_RESOLUTION, 2);

        if (!ok)
        {
            Serial.println("ERROR: ledcAttachChannel failed on PIN_GRIPPER");
            forceGripperPinLow();
            return false;
        }

        gripperInitialized = true;
    }

    gripperEnabled = true;

    uint32_t duty = pulseUsToDuty(gripperPulseUs);
    ledcWrite(PIN_GRIPPER, duty);

    delay(40);
    return true;
}

void disableGripperServo()
{
    if (!gripperEnabled)
    {
        if (!gripperInitialized) forceGripperPinLow();
        return;
    }

    // Poner duty a 0 silencia el servo sin desconectar el canal LEDC.
    // Hacer ledcDetach + ledcAttach dinámico mientras el buzzer task
    // escribe en otros canales LEDC desde el core 0 corrompe el estado
    // del periférico (race condition entre cores).
    ledcWrite(PIN_GRIPPER, 0);
    delay(5);

    gripperEnabled = false;
}

bool isGripperEnabled()
{
    return gripperEnabled;
}

// ===================== INICIALIZACIÓN =====================

void beginGripper()
{
    Serial.println("beginGripper LEDC: start");

    if (!enableGripperServo())
    {
        return;
    }

    openGripper();
    delay(GRIPPER_HOLD_OPEN_MS);

    disableGripperServo();

    Serial.println("beginGripper LEDC: end");
}

// ===================== MOVIMIENTO POR PULSO =====================

void moveGripperPulse(int pulseUs)
{
    pulseUs = constrain(pulseUs, 500, 2500);
    gripperPulseUs = pulseUs;

    if (!gripperEnabled)
    {
        if (!enableGripperServo())
        {
            return;
        }
    }

    uint32_t duty = pulseUsToDuty(pulseUs);
    ledcWrite(PIN_GRIPPER, duty);
}

void moveGripperSmoothPulse(int targetPulseUs)
{
    targetPulseUs = constrain(targetPulseUs, 500, 2500);

    if (!gripperEnabled)
    {
        if (!enableGripperServo())
        {
            return;
        }
    }

    int step = (targetPulseUs > gripperPulseUs) ? 10 : -10;

    for (int us = gripperPulseUs; us != targetPulseUs; us += step)
    {
        if ((step > 0 && us > targetPulseUs) ||
            (step < 0 && us < targetPulseUs))
        {
            break;
        }

        moveGripperPulse(us);
        delay(10);
    }

    moveGripperPulse(targetPulseUs);
}

// ===================== FUNCIONES BÁSICAS =====================

void openGripper()
{
    moveGripperPulse(GRIPPER_OPEN_US);
}

void closeGripper()
{
    moveGripperPulse(GRIPPER_CLOSE_US);
}

void openGripperSmooth()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_OPEN_US);
}

void closeGripperSmooth()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_CLOSE_US);
}

// ===================== FUNCIONES CON DETACH =====================

void closeGripperAndRelease()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_CLOSE_US);

    // Mantiene fuerza un instante para asegurar el agarre
    delay(GRIPPER_HOLD_CLOSE_MS);

    disableGripperServo();
}

void openGripperAndRelease()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_OPEN_US);

    // Espera para asegurar apertura
    delay(GRIPPER_HOLD_OPEN_MS);

    disableGripperServo();
}

void openGripperFromReleased()
{
    openGripperAndRelease();
}

// ===================== CONSULTA =====================

int getGripperPulseUs()
{
    return gripperPulseUs;
}