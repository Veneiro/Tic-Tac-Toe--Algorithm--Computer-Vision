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
static const uint8_t AMT_RESP_WAIT    = 0xA5;
static const uint8_t AMT_RESP_ZERO_OK = 0x80;

uint8_t SPI_T(uint8_t msg, uint8_t cs_pin)
{
	uint8_t resp;

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

	delayMicroseconds(100);

	return resp;
}

uint16_t readAMT203(uint8_t cs_pin)
{
	uint8_t received = 0x00;

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

void beginSPI(void)
{
	pinMode(PIN_CS1, OUTPUT);
	pinMode(PIN_CS2, OUTPUT);
	pinMode(PIN_CS3, OUTPUT);

	digitalWrite(PIN_CS1, HIGH);
	digitalWrite(PIN_CS2, HIGH);
	digitalWrite(PIN_CS3, HIGH);

	delay(100);

	AMT_spi.begin(PIN_SCLK, PIN_MISO, PIN_MOSI);

	delay(100);
}

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

void printAbsoluteEncoderRaw(void)
{
	Serial.print(" RAW1:");
	Serial.print(joint_encoder_raw[0]);

	Serial.print(" || RAW2:");
	Serial.print(joint_encoder_raw[1]);

	Serial.print(" || RAW3:");
	Serial.println(joint_encoder_raw[2]);
}

bool setZeroAMT203(uint8_t cs_pin)
{
	uint8_t resp = 0x00;

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
