#include <Arduino.h>
#include <SPI.h>
#include <math.h>

#include "config.h"
#include "encoder.h"

SPIClass AMT_spi(HSPI);

// ===================== CONFIGURACIÓN SPI =====================

// Mantén 250 kHz porque ya viste que es más estable en tu PCB.
// Luego puedes probar 500 kHz si todo va bien.
static const uint32_t AMT_SPI_FREQ = 250000;

// Número máximo de intentos, estilo código del profesor
static const int AMT_READ_MAX_TRIES = 80;
static const int AMT_ZERO_MAX_TRIES = 120;

// Valor especial para indicar error de lectura
static const uint16_t AMT_READ_ERROR = 0xFFFF;

// Comandos AMT203
static const uint8_t AMT_CMD_NOP      = 0x00;
static const uint8_t AMT_CMD_RD_POS   = 0x10;
static const uint8_t AMT_CMD_SET_ZERO = 0x70;

// Respuestas AMT203
static const uint8_t AMT_RESP_WAIT    = 0xA5;
static const uint8_t AMT_RESP_ZERO_OK = 0x80;


// ===================== TRANSFERENCIA SPI =====================

uint8_t SPI_T(uint8_t msg, uint8_t cs_pin)
{
    uint8_t resp;

    // Asegurar que ningún encoder quede seleccionado por accidente
    digitalWrite(PIN_CS1, HIGH);
    digitalWrite(PIN_CS2, HIGH);
    digitalWrite(PIN_CS3, HIGH);

    delayMicroseconds(5);

    AMT_spi.beginTransaction(SPISettings(AMT_SPI_FREQ, MSBFIRST, SPI_MODE0));

    digitalWrite(cs_pin, LOW);
    delayMicroseconds(5);

    resp = AMT_spi.transfer(msg);

    delayMicroseconds(5);
    digitalWrite(cs_pin, HIGH);

    AMT_spi.endTransaction();

    // Separación entre transacciones. Ya viste que tu bus necesita margen.
    delayMicroseconds(100);

    return resp;
}


// ===================== LECTURA AMT203 =====================
// Lógica adaptada del profesor:
// 1) Enviar 0x10
// 2) Mientras responda 0xA5, seguir mandando NOP
// 3) Cuando responda 0x10, leer MSB y LSB
// 4) Si no responde 0x10, devolver error

uint16_t readAMT203(uint8_t cs_pin)
{
    uint8_t received = 0x00;

    // Enviar comando de lectura. Ignoramos la respuesta de esta transferencia.
    SPI_T(AMT_CMD_RD_POS, cs_pin);

    for (int count = 0; count < AMT_READ_MAX_TRIES; count++)
    {
        delayMicroseconds(100);

        received = SPI_T(AMT_CMD_NOP, cs_pin);

        if (received == AMT_CMD_RD_POS)
        {
            uint8_t msb = SPI_T(AMT_CMD_NOP, cs_pin) & 0x0F;

            delayMicroseconds(100);

            uint8_t lsb = SPI_T(AMT_CMD_NOP, cs_pin);

            return ((uint16_t)msb << 8) | lsb;
        }
    }

    return AMT_READ_ERROR;
}


float readAngleDeg(uint8_t cs_pin)
{
    uint16_t pos = readAMT203(cs_pin);

    if (pos == AMT_READ_ERROR)
    {
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


// ===================== RESET ESTADO SOFTWARE =====================

void resetEncoderSoftwareState(void)
{
    for (int i = 0; i < 3; i++)
    {
        joint_encoder_raw_prev[i] = 0.0f;
        joint_encoder_raw[i]      = 0.0f;
        ramal_encoder[i]          = 0.0f;
        joint_encoder[i]          = 0.0f;
    }
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

        // Detección de cruce 0/360
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


// ===================== DEBUG RAW =====================

void printAbsoluteEncoderRaw(void)
{
    Serial.print(" RAW1:");
    Serial.print(joint_encoder_raw[0]);

    Serial.print(" || RAW2:");
    Serial.print(joint_encoder_raw[1]);

    Serial.print(" || RAW3:");
    Serial.println(joint_encoder_raw[2]);
}


// ===================== SET ZERO AMT203 =====================
// Lógica adaptada del profesor:
// 1) Enviar 0x70
// 2) Mandar NOP hasta recibir 0x80
// 3) Si recibe 0x80, el offset fue guardado en EEPROM
// 4) Luego debes apagar/encender el encoder para aplicar el nuevo cero

bool setZeroAMT203(uint8_t cs_pin)
{
    uint8_t resp = 0x00;

    // Enviar comando set_zero_point. Ignoramos la respuesta inmediata.
    SPI_T(AMT_CMD_SET_ZERO, cs_pin);

    for (int count = 0; count < AMT_ZERO_MAX_TRIES; count++)
    {
        delayMicroseconds(100);

        resp = SPI_T(AMT_CMD_NOP, cs_pin);

        if (resp == AMT_RESP_ZERO_OK)
        {
            return true;
        }
    }

    return false;
}


void setZeroAllEncoders(void)
{
    Serial.println("===== SET ZERO ENCODERS =====");

    Serial.println("Poniendo en cero encoder 1...");
    bool ok1 = setZeroAMT203(PIN_CS1);

    delay(100);

    Serial.println("Poniendo en cero encoder 2...");
    bool ok2 = setZeroAMT203(PIN_CS2);

    delay(100);

    Serial.println("Poniendo en cero encoder 3...");
    bool ok3 = setZeroAMT203(PIN_CS3);

    Serial.print("Zero E1: ");
    Serial.println(ok1 ? "OK" : "ERROR");

    Serial.print("Zero E2: ");
    Serial.println(ok2 ? "OK" : "ERROR");

    Serial.print("Zero E3: ");
    Serial.println(ok3 ? "OK" : "ERROR");

    resetEncoderSoftwareState();

    Serial.println("IMPORTANTE:");
    Serial.println("Si dice OK, el cero fue guardado en EEPROM.");
    Serial.println("Apaga y enciende la alimentacion de los encoders para aplicar el nuevo cero.");
    Serial.println("=============================");
}


// ===================== VERIFICAR CERO =====================

void verifyZeroAllEncoders(void)
{
    Serial.println("===== VERIFICANDO CERO =====");

    for (int i = 0; i < 3; i++)
    {
        float angle = readAngleDeg(pines_CS[i]);

        Serial.print("Encoder ");
        Serial.print(i + 1);
        Serial.print(": ");

        if (isnan(angle))
        {
            Serial.println("ERROR lectura");
            continue;
        }

        Serial.print(angle);
        Serial.print(" deg -> ");

        if (angle < 5.0f || angle > 355.0f)
        {
            Serial.println("CERCA DE CERO");
        }
        else
        {
            Serial.println("NO ESTA EN CERO");
        }

        delay(100);
    }

    resetEncoderSoftwareState();

    Serial.println("============================");
}
/*
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
    // Enviar comando set_zero_point
    SPI_T(AMT_CMD_SET_ZERO, cs_pin);

    unsigned long t0 = millis();
    uint8_t resp = AMT_RESP_WAIT;

    while (millis() - t0 < AMT_ZERO_TIMEOUT_MS)
    {
        delayMicroseconds(100);

        resp = SPI_T(AMT_CMD_NOP, cs_pin);

        if (resp == AMT_RESP_ZERO_OK)
        {
            // 0x80 significa que el offset fue guardado en EEPROM.
            // Aún falta apagar/encender el encoder para aplicar el nuevo cero.
            return true;
        }
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
*/