#ifndef MOTORDRIVER_H
#define MOTORDRIVER_H

void writeRegister(uint8_t reg, uint8_t *data, uint8_t len);
void readRegister(uint8_t reg, uint8_t *buffer, uint8_t len);
void initDriver(void);
void setMotorSpeed(float m1, float m2, float m3, float m4);
void setRobotJointPosition(AngularPosition targetPos);
void setMotorPosition2(float p1, float p2, float p3, float p4);
void setMotorPWM(int8_t m1, int8_t m2, int8_t m3, int8_t m4);
void readRelativeEncoders(void);
void beginWire(void);
void computeVelocity(void);
void printRelativeEncoder(void);

#endif