#ifndef KINEMATICS_H
#define KINEMATICS_H

float toEncoderFrame(float q_art, float q0, bool invertDirection);
float fromEncoderFrame(float q_enc, float q0, bool invertDirection);
float toEncoderFrameGeneral(float q_art, float E_ref, float A_ref, bool invertDirection);
float fromEncoderFrameGeneral(float q_enc, float E_ref, float A_ref, bool invertDirection);
float truncate(float x, float minVal, float maxVal);
IKResult inverseKinematics(LinearPosition targetPos);
void forwardKinematics(void);
float deg2rad(float deg);
float rad2deg(float rad);
void computeJointsAngle(void);
void printJoints(void);
void printPosition(void);
void printSetPointPosition(void);
void printSetPointJoints(void);
void pick(LinearPosition P_pick);
void place(LinearPosition P_place);
void returnHome(void);
void pickAndPlace(LinearPosition P_pick, LinearPosition P_place);
void test1(void);
void test2(void);

#endif
