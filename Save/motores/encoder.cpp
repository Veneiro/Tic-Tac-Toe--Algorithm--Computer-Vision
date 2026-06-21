#include <Arduino.h>
#include <SPI.h>
#include <math.h>

#include "config.h"
#include "encoder.h"

SPIClass AMT_spi(HSPI);

// ===================== CONFIGURACIÓN SPI =====================

// Según tu observación, prueba primero con 250 kHz.
// Si sigue fallando el encoder 3, baja a 100 kHz.
static const uint32_t AMT_SPI_FREQ = 250000;

// Timeout para esperar respuesta del encoder
static const uint32_t AMT_READ_TIMEOUT_MS = 50;
static const uint32_t AMT_ZERO_TIMEOUT_MS = 1000;

// Valor especial para indicar error de lectura
static const uint16_t AMT_READ_ERROR = 0xFFFF;

// Comandos AMT203
static const uint8_t AMT_CMD_NOP      = 0x00;
static const uint8_t AMT_CMD_RD_POS   = 0x10;
static const uint8_t AMT_CMD_SET_ZERO = 0x70;

static const uint8_t AMT_RESP_WAIT    = 0xA5;
static const uint8_t AMT_RESP_ZERO_OK = 0x80;


// ===================== TRANSFERENCIA SPI =====================

uint8_t SPI_T(uint8_t msg, uint8_t cs_pin)
{
    uint8_t resp;

    AMT_spi.beginTransaction(SPISettings(AMT_SPI_FREQ, MSBFIRST, SPI_MODE0));

    digitalWrite(cs_pin, LOW);
    delayMicroseconds(1);

    resp = AMT_spi.transfer(msg);

    delayMicroseconds(5);
    digitalWrite(cs_pin, HIGH);

    // El datasheet recomienda dejar separación entre lecturas;
    // 20 us es recomendado, aquí dejamos margen.
    delayMicroseconds(100);

    AMT_spi.endTransaction();

    return resp;
}


// ===================== LECTURA AMT203 =====================

uint16_t readAMT203(uint8_t cs_pin)
{
    uint8_t received = 0x00;
    uint8_t temp[2];

    // 1. Enviar comando de lectura de posición
    SPI_T(AMT_CMD_RD_POS, cs_pin);

    // 2. Enviar NOP hasta recibir eco 0x10
    unsigned long t0 = millis();

    do {
        received = SPI_T(AMT_CMD_NOP, cs_pin);

        if (millis() - t0 > AMT_READ_TIMEOUT_MS) {
            return AMT_READ_ERROR;
        }

    } while (received != AMT_CMD_RD_POS);

    // 3. Leer MSB y LSB
    temp[0] = SPI_T(AMT_CMD_NOP, cs_pin);
    temp[1] = SPI_T(AMT_CMD_NOP, cs_pin);

    // El byte MSB solo usa los 4 bits bajos
    temp[0] &= 0x0F;

    uint16_t ABSposition = ((uint16_t)temp[0] << 8) | temp[1];

    return ABSposition;
}


float readAngleDeg(uint8_t cs_pin)
{
    uint16_t pos = readAMT203(cs_pin);

    if (pos == AMT_READ_ERROR) {
        return NAN;
    }

    return (pos * 360.0f) / 4096.0f;
}


// ===================== INICIALIZACIÓN SPI =====================

void beginSPI(void)
{
    // Primero asegurar que todos los CS estén desactivados
    pinMode(PIN_CS1, OUTPUT);
    pinMode(PIN_CS2, OUTPUT);
    pinMode(PIN_CS3, OUTPUT);

    digitalWrite(PIN_CS1, HIGH);
    digitalWrite(PIN_CS2, HIGH);
    digitalWrite(PIN_CS3, HIGH);

    delay(100);

    // Luego iniciar SPI
    AMT_spi.begin(PIN_SCLK, PIN_MISO, PIN_MOSI);

    delay(100);
}


// ===================== LECTURA DE TODOS LOS ENCODERS =====================

void readAbsoluteEncoder(void)
{
    for (int i = 0; i < 3; i++)
    {
        float angle = readAngleDeg(pines_CS[i]);

        if (isnan(angle))
        {
            Serial.print("ERROR lectura encoder ");
            Serial.println(i + 1);

            // No actualizamos ese encoder si falló
            continue;
        }

        joint_encoder_raw_prev[i] = joint_encoder_raw[i];
        joint_encoder_raw[i] = angle;

        float dif = joint_encoder_raw[i] - joint_encoder_raw_prev[i];

        // Detección de cruce 0/360.
        // Mejor usar 180° que 90° para evitar falsos saltos.
        if (dif > 180.0f)
        {
            ramal_encoder[i] -= 360.0f;
        }
        else if (dif < -180.0f)
        {
            ramal_encoder[i] += 360.0f;
        }

        joint_encoder[i] = joint_encoder_raw[i] + ramal_encoder[i];
    }
}


void printAbsoluteEncoder(void)
{
    Serial.print(" AE1:");
    Serial.print(joint_encoder[0]);

    Serial.print(" || AE2:");
    Serial.print(joint_encoder[1]);

    Serial.print(" || AE3:");
    Serial.println(joint_encoder[2]);
}


// ===================== SET ZERO AMT203 =====================

bool setZeroAMT203(uint8_t cs_pin)
{
    // 1. Enviar comando set_zero_point
    SPI_T(AMT_CMD_SET_ZERO, cs_pin);

    // 2. Enviar NOP hasta recibir 0x80
    unsigned long t0 = millis();
    uint8_t resp = AMT_RESP_WAIT;

    while (millis() - t0 < AMT_ZERO_TIMEOUT_MS)
    {
        resp = SPI_T(AMT_CMD_NOP, cs_pin);

        if (resp == AMT_RESP_ZERO_OK)
        {
            return true;
        }

        delayMicroseconds(100);
    }

    return false;
}


void setZeroAllEncoders(void)
{
    Serial.println("Poniendo en cero encoder 1...");
    bool ok1 = setZeroAMT203(PIN_CS1);

    Serial.println("Poniendo en cero encoder 2...");
    bool ok2 = setZeroAMT203(PIN_CS2);

    Serial.println("Poniendo en cero encoder 3...");
    bool ok3 = setZeroAMT203(PIN_CS3);

    Serial.print("Zero E1: ");
    Serial.println(ok1 ? "OK" : "ERROR");

    Serial.print("Zero E2: ");
    Serial.println(ok2 ? "OK" : "ERROR");

    Serial.print("Zero E3: ");
    Serial.println(ok3 ? "OK" : "ERROR");

    Serial.println("IMPORTANTE: Apaga y enciende la alimentacion de los encoders para aplicar el nuevo cero.");
}