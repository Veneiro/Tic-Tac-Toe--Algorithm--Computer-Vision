#include <Arduino.h>
#include "config.h"
#include "gripper.h"

const int GRIPPER_OPEN_US  = 1300;
const int GRIPPER_CLOSE_US = 1950;

const int SERVO_FREQ = 50;
const int SERVO_RESOLUTION = 14;
const int SERVO_CHANNEL = 0;

const int GRIPPER_HOLD_CLOSE_MS = 400;
const int GRIPPER_HOLD_OPEN_MS = 300;

static int gripperPulseUs = GRIPPER_OPEN_US;
static bool gripperEnabled = false;

static uint32_t pulseUsToDuty(int pulseUs)
{
	pulseUs = constrain(pulseUs, 500, 2500);

	uint32_t maxDuty = (1UL << SERVO_RESOLUTION) - 1;

	return (uint32_t)((pulseUs / 20000.0f) * maxDuty);
}

static void writeGripperDuty(uint32_t duty)
{
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
	ledcWrite(PIN_GRIPPER, duty);
#else
	ledcWrite(SERVO_CHANNEL, duty);
#endif
}

static bool attachGripperServo()
{
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
	return ledcAttach(PIN_GRIPPER, SERVO_FREQ, SERVO_RESOLUTION);
#else
	double actualFreq = ledcSetup(SERVO_CHANNEL, SERVO_FREQ, SERVO_RESOLUTION);
	if (actualFreq <= 0.0) {
		return false;
	}

	ledcAttachPin(PIN_GRIPPER, SERVO_CHANNEL);
	return true;
#endif
}

static bool detachGripperServo()
{
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
	return ledcDetach(PIN_GRIPPER);
#else
	ledcDetachPin(PIN_GRIPPER);
	return true;
#endif
}

bool enableGripperServo()
{
	if (gripperEnabled) {
		return true;
	}

	bool ok = attachGripperServo();

	if (!ok) {
		Serial.println("ERROR: ledcAttach fallo en PIN_GRIPPER");
		gripperEnabled = false;
		return false;
	}

	gripperEnabled = true;

	uint32_t duty = pulseUsToDuty(gripperPulseUs);
	writeGripperDuty(duty);

	return true;
}

void disableGripperServo()
{
	if (!gripperEnabled) {
		return;
	}

	writeGripperDuty(0);
	delay(20);

	bool ok = detachGripperServo();

	if (!ok) {
		Serial.println("ADVERTENCIA: ledcDetach fallo en PIN_GRIPPER");
	}

	pinMode(PIN_GRIPPER, OUTPUT);
	digitalWrite(PIN_GRIPPER, LOW);

	gripperEnabled = false;
}

bool isGripperEnabled()
{
	return gripperEnabled;
}

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
	writeGripperDuty(duty);
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

void closeGripperAndRelease()
{
	enableGripperServo();
	moveGripperSmoothPulse(GRIPPER_CLOSE_US);
	delay(GRIPPER_HOLD_CLOSE_MS);
	disableGripperServo();
}

void openGripperAndRelease()
{
	enableGripperServo();
	moveGripperSmoothPulse(GRIPPER_OPEN_US);
	delay(GRIPPER_HOLD_OPEN_MS);
	disableGripperServo();
}

void openGripperFromReleased()
{
	openGripperAndRelease();
}

int getGripperPulseUs()
{
	return gripperPulseUs;
}
