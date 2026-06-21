#include <Arduino.h>
#include "config.h"
#include "gripper.h"

// ===================== CONFIGURACIÓN DE PINZA =====================

const int GRIPPER_OPEN_US  = 1300;
const int GRIPPER_CLOSE_US = 1950;

const int SERVO_FREQ = 50;
const int SERVO_RESOLUTION = 14;

// Tiempo que mantiene fuerza después de cerrar.
// Ajusta entre 200 y 800 ms según la pieza.
const int GRIPPER_HOLD_CLOSE_MS = 400;

// Tiempo para asegurar que abrió antes de desvincular.
// Puede ser menor que el cierre porque no necesita hacer presión.
const int GRIPPER_HOLD_OPEN_MS = 300;

static int gripperPulseUs = GRIPPER_OPEN_US;
static bool gripperEnabled = false;

// ===================== FUNCIONES INTERNAS =====================

static uint32_t pulseUsToDuty(int pulseUs)
{
    pulseUs = constrain(pulseUs, 500, 2500);

    uint32_t maxDuty = (1UL << SERVO_RESOLUTION) - 1;

    // 50 Hz -> periodo = 20 000 us
    return (uint32_t)((pulseUs / 20000.0f) * maxDuty);
}

// ===================== CONTROL DE ACTIVACIÓN =====================

bool enableGripperServo()
{
    if (gripperEnabled) {
        return true;
    }

    bool ok = ledcAttach(PIN_GRIPPER, SERVO_FREQ, SERVO_RESOLUTION);

    if (!ok) {
        Serial.println("ERROR: ledcAttach fallo en PIN_GRIPPER");
        gripperEnabled = false;
        return false;
    }

    gripperEnabled = true;

    // Recupera el último pulso conocido al reactivar
    uint32_t duty = pulseUsToDuty(gripperPulseUs);
    ledcWrite(PIN_GRIPPER, duty);

    return true;
}

void disableGripperServo()
{
    if (!gripperEnabled) {
        return;
    }

    // Deja de enviar pulso PWM
    ledcWrite(PIN_GRIPPER, 0);
    delay(20);

    bool ok = ledcDetach(PIN_GRIPPER);

    if (!ok) {
        Serial.println("ADVERTENCIA: ledcDetach fallo en PIN_GRIPPER");
    }

    // Deja el pin en bajo para evitar señal flotante
    pinMode(PIN_GRIPPER, OUTPUT);
    digitalWrite(PIN_GRIPPER, LOW);

    gripperEnabled = false;
}

bool isGripperEnabled()
{
    return gripperEnabled;
}

// ===================== INICIALIZACIÓN =====================

void beginGripper()
{
    Serial.println("beginGripper LEDC: inicio");

    if (!enableGripperServo()) {
        return;
    }

    openGripper();
    delay(GRIPPER_HOLD_OPEN_MS);
    disableGripperServo();

    Serial.println("beginGripper LEDC: fin");
}

// ===================== MOVIMIENTO POR PULSO =====================

void moveGripperPulse(int pulseUs)
{
    pulseUs = constrain(pulseUs, 500, 2500);
    gripperPulseUs = pulseUs;

    if (!gripperEnabled) {
        if (!enableGripperServo()) {
            return;
        }
    }

    uint32_t duty = pulseUsToDuty(pulseUs);
    ledcWrite(PIN_GRIPPER, duty);
}

void moveGripperSmoothPulse(int targetPulseUs)
{
    targetPulseUs = constrain(targetPulseUs, 500, 2500);

    if (!gripperEnabled) {
        if (!enableGripperServo()) {
            return;
        }
    }

    int step = (targetPulseUs > gripperPulseUs) ? 10 : -10;

    for (int us = gripperPulseUs; us != targetPulseUs; us += step)
    {
        if ((step > 0 && us > targetPulseUs) || 
            (step < 0 && us < targetPulseUs)) {
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
    enableGripperServo();
    moveGripperSmoothPulse(GRIPPER_OPEN_US);
}

void closeGripperSmooth()
{
    enableGripperServo();
    moveGripperSmoothPulse(GRIPPER_CLOSE_US);
}

// ===================== FUNCIONES CON DETACH =====================

void closeGripperAndRelease()
{
    enableGripperServo();

    // Cierra con fuerza hasta el pulso definido
    moveGripperSmoothPulse(GRIPPER_CLOSE_US);

    // Mantiene fuerza un instante para asegurar el agarre
    delay(GRIPPER_HOLD_CLOSE_MS);

    // Deja de enviar PWM para reducir consumo
    disableGripperServo();
}

void openGripperAndRelease()
{
    enableGripperServo();

    // Abre la pinza
    moveGripperSmoothPulse(GRIPPER_OPEN_US);

    // Espera para asegurar que llegó a la posición abierta
    delay(GRIPPER_HOLD_OPEN_MS);

    // Deja de enviar PWM para reducir consumo
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