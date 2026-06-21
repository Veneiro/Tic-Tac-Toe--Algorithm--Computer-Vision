#include <Arduino.h>
#include <SPI.h>
#include <math.h>

#include "config.h"
#include "encoder.h"

SPIClass AMT_spi(HSPI);


static const uint32_t AMT_SPI_FREQ = 250000;

static const int AMT_READ_MAX_TRIES = 80;
static const int AMT_ZERO_MAX_TRIES = 120;

static const uint16_t AMT_READ_ERROR = 0xFFFF;


static const uint8_t AMT_CMD_NOP      = 0x00;
static const uint8_t AMT_CMD_RD_POS   = 0x10;
static const uint8_t AMT_CMD_SET_ZERO = 0x70;


static const uint8_t AMT_RESP_ZERO_OK = 0x80;


static const int AMT_VALIDATION_DIFF_COUNTS = 8;   // 8 cuentas ≈ 0.70 grados


/**
 * @brief Pone los tres pines CS en HIGH para deseleccionar todos los encoders.
 */
static void deselectAllEncoders()
{
    digitalWrite(PIN_CS1, HIGH);
    digitalWrite(PIN_CS2, HIGH);
    digitalWrite(PIN_CS3, HIGH);
}


/**
 * @brief Calcula la diferencia circular entre dos lecturas del encoder (rango 0-4095).
 * @param a Valor A en cuentas (0-4095).
 * @param b Valor B en cuentas (0-4095).
 * @return Diferencia con signo en el rango [-2048, 2048], teniendo en cuenta el cruce 0/360.
 */
static int circularDiffCounts(uint16_t a, uint16_t b)
{
    int diff = (int)a - (int)b;

    if (diff > 2048)
    {
        diff -= 4096;
    }
    else if (diff < -2048)
    {
        diff += 4096;
    }

    return diff;
}


/**
 * @brief Calcula la media circular entre dos lecturas del encoder.
 * @param a Valor A en cuentas (0-4095).
 * @param b Valor B en cuentas (0-4095).
 * @return Media circular en cuentas (0-4095).
 */
static uint16_t circularAverageCounts(uint16_t a, uint16_t b)
{
    int diff = circularDiffCounts(b, a);
    int avg = (int)a + diff / 2;

    while (avg < 0)
    {
        avg += 4096;
    }

    while (avg >= 4096)
    {
        avg -= 4096;
    }

    return (uint16_t)avg;
}


/**
 * @brief Realiza una transferencia SPI de un byte al encoder seleccionado.
 * @param msg    Byte a enviar.
 * @param cs_pin Pin de chip select del encoder destino.
 * @return Byte recibido del encoder durante la transferencia.
 */
uint8_t SPI_T(uint8_t msg, uint8_t cs_pin)
{
    uint8_t resp;

    deselectAllEncoders();

    delayMicroseconds(10);

    AMT_spi.beginTransaction(SPISettings(AMT_SPI_FREQ, MSBFIRST, SPI_MODE0));

    digitalWrite(cs_pin, LOW);
    delayMicroseconds(10);

    resp = AMT_spi.transfer(msg);

    delayMicroseconds(10);
    digitalWrite(cs_pin, HIGH);

    AMT_spi.endTransaction();

    delayMicroseconds(150);

    return resp;
}


/**
 * @brief Lee la posición absoluta del encoder AMT203S-V siguiendo su protocolo SPI.
 * @param cs_pin Pin de chip select del encoder a leer.
 * @return Posición en cuentas (0-4095), o AMT_READ_ERROR (0xFFFF) si falla.
 */
uint16_t readAMT203(uint8_t cs_pin)
{
    uint8_t received = 0x00;

    SPI_T(AMT_CMD_RD_POS, cs_pin);

    for (int count = 0; count < AMT_READ_MAX_TRIES; count++)
    {
        delayMicroseconds(150);

        received = SPI_T(AMT_CMD_NOP, cs_pin);

        if (received == AMT_CMD_RD_POS)
        {
            delayMicroseconds(150);

            uint8_t msb = SPI_T(AMT_CMD_NOP, cs_pin) & 0x0F;

            delayMicroseconds(150);

            uint8_t lsb = SPI_T(AMT_CMD_NOP, cs_pin);

            uint16_t pos = ((uint16_t)msb << 8) | lsb;

            if (pos <= 4095)
            {
                return pos;
            }

            return AMT_READ_ERROR;
        }
    }

    return AMT_READ_ERROR;
}


/**
 * @brief Lee la posición del encoder tres veces y valida por mayoría.
 *        Acepta el resultado solo si al menos dos lecturas coinciden dentro de AMT_VALIDATION_DIFF_COUNTS.
 * @param cs_pin Pin de chip select del encoder a leer.
 * @return Media de las dos lecturas coherentes, o AMT_READ_ERROR si ningún par concuerda.
 */
uint16_t readAMT203Validated(uint8_t cs_pin)
{
    uint16_t r1 = readAMT203(cs_pin);
    delay(2);

    uint16_t r2 = readAMT203(cs_pin);
    delay(2);

    uint16_t r3 = readAMT203(cs_pin);

    if (r1 == AMT_READ_ERROR || r2 == AMT_READ_ERROR || r3 == AMT_READ_ERROR)
    {
        return AMT_READ_ERROR;
    }

    int d12 = abs(circularDiffCounts(r1, r2));
    int d13 = abs(circularDiffCounts(r1, r3));
    int d23 = abs(circularDiffCounts(r2, r3));

    if (d12 <= AMT_VALIDATION_DIFF_COUNTS)
    {
        return circularAverageCounts(r1, r2);
    }

    if (d13 <= AMT_VALIDATION_DIFF_COUNTS)
    {
        return circularAverageCounts(r1, r3);
    }

    if (d23 <= AMT_VALIDATION_DIFF_COUNTS)
    {
        return circularAverageCounts(r2, r3);
    }

    return AMT_READ_ERROR;
}


/**
 * @brief Lee la posición del encoder y la convierte a grados (0-360).
 * @param cs_pin Pin de chip select del encoder a leer.
 * @return Ángulo en grados (0.0-360.0), o NAN si la lectura falla.
 */
float readAngleDeg(uint8_t cs_pin)
{
    uint16_t pos = readAMT203Validated(cs_pin);

    if (pos == AMT_READ_ERROR)
    {
        return NAN;
    }

    return (pos * 360.0f) / 4096.0f;
}


/**
 * @brief Envía 10 NOPs al encoder para limpiar su buffer SPI interno.
 * @param cs_pin Pin de chip select del encoder a vaciar.
 */
void flushEncoder(uint8_t cs_pin)
{
    for (int k = 0; k < 10; k++)
    {
        SPI_T(AMT_CMD_NOP, cs_pin);
        delayMicroseconds(300);
    }
}


/**
 * @brief Llama a flushEncoder sobre los tres encoders del sistema.
 */
void flushAllEncoders()
{
    flushEncoder(PIN_CS1);
    flushEncoder(PIN_CS2);
    flushEncoder(PIN_CS3);
}


/**
 * @brief Inicializa el bus SPI y los pines CS de los tres encoders AMT203S-V.
 */
void beginSPI(void)
{
    pinMode(PIN_CS1, OUTPUT);
    pinMode(PIN_CS2, OUTPUT);
    pinMode(PIN_CS3, OUTPUT);

    deselectAllEncoders();

    delay(300);

    AMT_spi.begin(PIN_SCLK, PIN_MISO, PIN_MOSI);

    delay(300);

    flushAllEncoders();

    delay(100);
}


/**
 * @brief Pone a cero las variables software de seguimiento de ángulo de los tres ejes.
 */
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


/**
 * @brief Lee los tres encoders absolutos y actualiza joint_encoder[] con seguimiento de vueltas.
 *        Detecta cruces 0/360 para mantener continuidad en el ángulo acumulado.
 */
void readAbsoluteEncoder(void)
{
    for (int i = 0; i < 3; i++)
    {
        float angle = readAngleDeg(pines_CS[i]);

        if (isnan(angle))
        {
            Serial.print("ERROR lectura encoder ");
            Serial.println(i + 1);

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


/**
 * @brief Imprime por Serial los ángulos acumulados de los tres encoders (joint_encoder[]).
 */
void printAbsoluteEncoder(void)
{
    Serial.print(" AE1:");
    Serial.print(joint_encoder[0]);

    Serial.print(" || AE2:");
    Serial.print(joint_encoder[1]);

    Serial.print(" || AE3:");
    Serial.println(joint_encoder[2]);
}


/**
 * @brief Imprime por Serial los valores crudos (sin seguimiento de vueltas) de los tres encoders.
 */
void printAbsoluteEncoderRaw(void)
{
    Serial.print(" RAW1:");
    Serial.print(joint_encoder_raw[0]);

    Serial.print(" || RAW2:");
    Serial.print(joint_encoder_raw[1]);

    Serial.print(" || RAW3:");
    Serial.println(joint_encoder_raw[2]);
}


/**
 * @brief Envía el comando de set-zero al encoder y espera la confirmación.
 *        El nuevo cero se graba en la EEPROM interna del encoder.
 * @param cs_pin Pin de chip select del encoder a resetear.
 * @return true si el encoder confirmó la operación (0x80), false si se agotaron los reintentos.
 */
bool setZeroAMT203(uint8_t cs_pin)
{
    uint8_t resp = 0x00;

    SPI_T(AMT_CMD_SET_ZERO, cs_pin);

    for (int count = 0; count < AMT_ZERO_MAX_TRIES; count++)
    {
        delayMicroseconds(150);

        resp = SPI_T(AMT_CMD_NOP, cs_pin);

        if (resp == AMT_RESP_ZERO_OK)
        {
            return true;
        }
    }

    return false;
}


/**
 * @brief Ejecuta el set-zero en los tres encoders secuencialmente e imprime el resultado.
 */
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


/**
 * @brief Lee los tres encoders e indica por Serial si cada uno está cerca del cero (< 5°).
 */
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
