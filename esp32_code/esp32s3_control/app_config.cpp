#include "app_contracts.h"

//const char *ssid = "MIM-DPM-GRUPO-3";
//const char *password = "mim-dpm-2026";
const char *ssid = "Galaxy S23 E57C";
const char *password = "2ppyahxwjx4g7zu";

const char *raspberryPi_IP = "10.191.81.8";
const int raspberryPi_PORT = 5000;
const bool reenviarTableroRaspberry = true;
// Use an explicit IP for the ESP32-CAM to avoid mDNS resolution issues.
// Replace the IP below with your camera's IP on the network.
const char *esp32CamURL = "http://10.191.81.88/capturar";

WebServer server(80);

const int pinJoyX = 2;
const int pinJoyY = 1;
const int pinJoyButton = 3;

Joystick joystick(
	pinJoyX,
	pinJoyY,
	pinJoyButton,
	0, 2022, 4096,
	0, 1968, 4096);

BoxWorkspace workspace = {
	-10.0f, 9.0f,
	24.0f, 40.0f,
	z_trabajo, 16.0f
};

LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);

const int pinLedRojo = 4;
const int pinLedAmarillo = 5;
const int pinLedVerde = 6;

const int pinSwitch = 36;
const int pinStart = 35;
const int pinMenu = 42;

const int pinSDA = 8;
const int pinSCL = 9;
