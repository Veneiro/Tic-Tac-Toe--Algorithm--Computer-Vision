#ifndef ENCODER_H
#define ENCODER_H

#include <Arduino.h>
#include <stdint.h>

void beginSPI(void);

uint8_t SPI_T(uint8_t msg, uint8_t cs_pin);

uint16_t readAMT203(uint8_t cs_pin);
uint16_t readAMT203Validated(uint8_t cs_pin);

float readAngleDeg(uint8_t cs_pin);

void flushEncoder(uint8_t cs_pin);
void flushAllEncoders(void);

void readAbsoluteEncoder(void);
void printAbsoluteEncoder(void);
void printAbsoluteEncoderRaw(void);

bool setZeroAMT203(uint8_t cs_pin);
void setZeroAllEncoders(void);

void resetEncoderSoftwareState(void);
void verifyZeroAllEncoders(void);

#endif
/*
#ifndef ENCODER_H
#define ENCODER_H

#include <Arduino.h>
#include <SPI.h>

extern SPIClass AMT_spi;

void beginSPI(void);

uint8_t SPI_T(uint8_t msg, uint8_t cs_pin);

uint16_t readAMT203(uint8_t cs_pin);
float readAngleDeg(uint8_t cs_pin);

void readAbsoluteEncoder(void);
void printAbsoluteEncoder(void);

bool setZeroAMT203(uint8_t cs_pin);
void setZeroAllEncoders(void);

#endif
*/