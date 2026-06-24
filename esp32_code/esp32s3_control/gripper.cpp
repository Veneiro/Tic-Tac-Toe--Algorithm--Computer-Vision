#include <Arduino.h>
#include "config.h"
#include "gripper.h"

const int GRIPPER_OPEN_US  = 1200;
const int GRIPPER_CLOSE_US = 1905;

const int SERVO_FREQ = 50;
const int SERVO_RESOLUTION = 14;

const int GRIPPER_HOLD_CLOSE_MS = 400;
const int GRIPPER_HOLD_OPEN_MS  = 300;

static int gripperPulseUs = GRIPPER_OPEN_US;
static bool gripperEnabled     = false;
static bool gripperInitialized = false;

/** @brief Convierte un ancho de pulso del servo en microsegundos a un ciclo de trabajo LEDC de 14 bits.
 *  Asume una señal de 50 Hz (periodo de 20 000 µs). La entrada se limita a [500, 2500] µs.
 *  @param pulseUs Ancho de pulso deseado en microsegundos.
 *  @return Valor de ciclo de trabajo LEDC para el canal de 14 bits. */
static uint32_t pulseUsToDuty(int pulseUs)
{
    pulseUs = constrain(pulseUs, 500, 2500);

    uint32_t maxDuty = (1UL << SERVO_RESOLUTION) - 1;

    // 50 Hz -> periodo = 20 000 us
    return (uint32_t)((pulseUs / 20000.0f) * maxDuty);
}

/** @brief Fuerza PIN_GRIPPER a nivel bajo como salida digital para establecer un estado seguro antes de inicializar LEDC. */
static void forceGripperPinLow()
{
    pinMode(PIN_GRIPPER, OUTPUT);
    digitalWrite(PIN_GRIPPER, LOW);
}


/** @brief Conecta el canal LEDC 2 (una sola vez) y habilita la salida del servo.
 *  Escribe el último ciclo de trabajo conocido para que el servo se mueva a su posición almacenada.
 *  @return Verdadero si el servo se habilitó correctamente; falso si falló la conexión LEDC. */
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

/** @brief Establece el ciclo de trabajo LEDC a 0, silenciando el servo sin desconectar el canal LEDC.
 *  Esto evita una condición de carrera entre núcleos con la tarea del buzzer en el núcleo 0. */
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

/** @brief Devuelve si el servo de la pinza está actualmente habilitado (emitiendo señal PWM).
 *  @return Verdadero si el servo está habilitado; falso en caso contrario. */
bool isGripperEnabled()
{
    return gripperEnabled;
}


/** @brief Inicializa la pinza: habilita el servo, la abre, mantiene brevemente la posición y luego la deshabilita. */
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


/** @brief Mueve la pinza inmediatamente al ancho de pulso especificado.
 *  Habilita el servo si actualmente está deshabilitado.
 *  @param pulseUs Ancho de pulso objetivo en microsegundos (limitado a [500, 2500]). */
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

/** @brief Desplaza la pinza desde su posición actual hasta el objetivo en pasos de 10 µs cada 10 ms.
 *  Habilita el servo automáticamente si no está activo.
 *  @param targetPulseUs Ancho de pulso final deseado en microsegundos (limitado a [500, 2500]). */
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


/** @brief Mueve la pinza inmediatamente a la posición completamente abierta (GRIPPER_OPEN_US). */
void openGripper()
{
    moveGripperPulse(GRIPPER_OPEN_US);
}

/** @brief Mueve la pinza inmediatamente a la posición completamente cerrada (GRIPPER_CLOSE_US). */
void closeGripper()
{
    moveGripperPulse(GRIPPER_CLOSE_US);
}

/** @brief Habilita el servo y mueve la pinza suavemente a la posición completamente abierta. */
void openGripperSmooth()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_OPEN_US);
}

/** @brief Habilita el servo y mueve la pinza suavemente a la posición completamente cerrada. */
void closeGripperSmooth()
{
    if (!enableGripperServo())
    {
        return;
    }

    moveGripperSmoothPulse(GRIPPER_CLOSE_US);
}


/** @brief Cierra la pinza suavemente, mantiene durante GRIPPER_HOLD_CLOSE_MS y luego deshabilita el servo. */
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

/** @brief Abre la pinza suavemente, mantiene durante GRIPPER_HOLD_OPEN_MS y luego deshabilita el servo. */
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

/** @brief Alias de openGripperAndRelease: abre la pinza desde un estado liberado (servo apagado). */
void openGripperFromReleased()
{
    openGripperAndRelease();
}


/** @brief Devuelve el ancho de pulso actual de la pinza en microsegundos.
 *  @return Último ancho de pulso establecido (µs), que refleja la posición actual o última del servo. */
int getGripperPulseUs()
{
    return gripperPulseUs;
}