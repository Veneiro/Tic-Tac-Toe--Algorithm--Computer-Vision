#include <Arduino.h>
#include "app_contracts.h"
#include "config.h"
#include "kinematics.h"
#include "gripper.h"

namespace
{
	void mantenerPantallaTurnoRobot()
	{
		if (modoAutomatico && turnoMaquina)
		{
			dibujarDecoracionTurnoAuto();
		}
	}
}

void computeJointsAngle(void){
  my_robot.q.q1 = fromEncoderFrameGeneral(joint_encoder[0],encoder_referencia_calibracion[0],joint_referencia_calibracion[0],false);
  my_robot.q.q2 = fromEncoderFrameGeneral(joint_encoder[1],encoder_referencia_calibracion[1],joint_referencia_calibracion[1],false);
  my_robot.q.q3 = fromEncoderFrameGeneral(joint_encoder[2],encoder_referencia_calibracion[2],joint_referencia_calibracion[2],true);
}

float toEncoderFrame(float q_art, float q0, bool invertDirection){
	float q_enc;

	if(invertDirection)
		q_enc = -q_art + q0;
	else
		q_enc =  q_art + q0;

	return q_enc;
}

float fromEncoderFrame(float q_enc, float q0, bool invertDirection){
	float q_art;

	if(invertDirection)
		q_art = -(q_enc - q0);
	else
		q_art =  (q_enc - q0);

	return q_art;
}

float toEncoderFrameGeneral(float q_art, float E_ref, float A_ref, bool invertDirection){
	float delta_art = q_art - A_ref;
	float q_enc;

	if (invertDirection)
		q_enc = E_ref - delta_art;
	else
		q_enc = E_ref + delta_art;

	return q_enc;
}

float fromEncoderFrameGeneral(float q_enc, float E_ref, float A_ref, bool invertDirection){
	float delta_enc = q_enc - E_ref;
	float q_art;

	if (invertDirection)
		q_art = A_ref - delta_enc;
	else
		q_art = A_ref + delta_enc;

	return q_art;
}

float truncate(float x, float minVal, float maxVal){
	if (x < minVal)
		return minVal;
	if (x > maxVal)
		return maxVal;
	return x;
}

IKResult inverseKinematics(LinearPosition targetPos){
	float X = targetPos.x;
	float Y = targetPos.y;
	float Z = targetPos.z;
	IKResult resultado;
	resultado.hasSolution = false;
	resultado.withinLimits = false;
	float q1 = atan2(-X, Y);

	float r  = sqrt(X * X + Y * Y);
	float zp = Z + L3;

	float D = (L1 * L1 + L2 * L2 - r * r - zp * zp) / (2.0 * L1 * L2);

	if (D < -1.0 || D > 1.0) {
		return resultado;
	}
	resultado.hasSolution = true;

	float q3 = atan2(D, sqrt(1.0 - D * D));

	float A = L1 - L2 * sin(q3);
	float B = L2 * cos(q3);

	float q2 = atan2(r, zp) - atan2(B, A);

	q1 = rad2deg(q1);
	q2 = rad2deg(q2);
	q3 = rad2deg(q3);

	if (q1 < Q1_MIN || q1 > Q1_MAX || q2 < Q2_MIN || q2 > Q2_MAX || q3 < Q3_MIN || q3 > Q3_MAX) {
	  q1 = truncate(q1, Q1_MIN, Q1_MAX);
	  q2 = truncate(q2, Q2_MIN, Q2_MAX);
	  q3 = truncate(q3, Q3_MIN, Q3_MAX);
	  resultado.q.q1 = q1;
	  resultado.q.q2 = q2;
	  resultado.q.q3 = q3;
	  return resultado;
	}
	resultado.q.q1 = q1;
	resultado.q.q2 = q2;
	resultado.q.q3 = q3;
	resultado.withinLimits = true;
	return resultado;
}

void forwardKinematics(void){

	float q1 = deg2rad(my_robot.q.q1);
	float q2 = deg2rad(my_robot.q.q2);
	float q3 = deg2rad(my_robot.q.q3);

	float sigma1 = L2 * cos(q2 + q3) + L1 * sin(q2);

	float X = -sin(q1) * sigma1;
	float Y =  cos(q1) * sigma1;
	float Z =  L1 * cos(q2) - L2 * sin(q2 + q3) - L3;
	my_robot.p.x = X;
	my_robot.p.y = Y;
	my_robot.p.z = Z;
}

float deg2rad(float deg){
	return deg * PI / 180.0;
}

float rad2deg(float rad){
	return rad * 180.0 / PI;
}

void printJoints(void){
	Serial.print(" q1:");
	Serial.print(my_robot.q.q1);
	Serial.print("|| q2:");
	Serial.print(my_robot.q.q2);
	Serial.print("|| q3:");
	Serial.println(my_robot.q.q3);
}

void printPosition(void){
	Serial.print(" x:");
	Serial.print(my_robot.p.x);
	Serial.print("|| y:");
	Serial.print(my_robot.p.y);
	Serial.print("|| z:");
	Serial.println(my_robot.p.z);
}

void printSetPointPosition(void){
	Serial.print(" SP_x:");
	Serial.print(target_position.x);
	Serial.print(" SP_y:");
	Serial.print(target_position.y);
	Serial.print(" SP_z:");
	Serial.println(target_position.z);
}

void printSetPointJoints(void){
	Serial.print(" SP_q1:");
	Serial.print(target_angle.q1);
	Serial.print(" SP_q2:");
	Serial.print(target_angle.q2);
	Serial.print(" SP_q3:");
	Serial.println(target_angle.q3);
}

void pick(LinearPosition P_pick){
	LinearPosition P0 = my_robot.p;
	LinearPosition P1 = {P_pick.x, P_pick.y, z_elevacion};
	LinearPosition P2 = {P_pick.x, P_pick.y, z_trabajo};

	LinearPosition T0 = {0.0f, 0.0f, -20.0f};
	LinearPosition T1 = {0.0f, 0.0f, -20.0f};

	LinearPosition T2 = {0.0f, 0.0f, -20.0f};
	LinearPosition T3 = {0.0f, 0.0f, -20.0f};

	HermiteSegment seg1(P0, P1, T0, T1);
	HermiteSegment seg2(P1, P2, T2, T3);

	TrajectoryPath path;
	path.addSegment(seg1);
	path.addSegment(seg2);
	path.build();
	float S = path.getTotalLength();
	SinusoidalSCurveProfile profile(S, V_max, T_acc);
	float Ttotal = profile.getTotalTime();
	float t = 0.0;
	while(t<Ttotal){
		MotionState ms = profile.evaluate(t);
		target_position = path.evaluateByDistance(ms.s);
		IKResult my_solution = inverseKinematics(target_position);
		ok = my_solution.hasSolution;
		target_angle = my_solution.q;
		t+=Ts;
		mantenerPantallaTurnoRobot();
		delay(50);
	}
	MotionState ms = profile.evaluate(Ttotal);
	target_position = path.evaluateByDistance(ms.s);
	Serial.print("final s1: ");
	printSetPointPosition();
}

void place(LinearPosition P_place){
	LinearPosition P0 = my_robot.p;
	LinearPosition P1 = {my_robot.p.x, my_robot.p.y, z_elevacion};
	LinearPosition P2 = {P_place.x, P_place.y, z_elevacion};
	LinearPosition P3 = {P_place.x, P_place.y, z_trabajo};

	LinearPosition T0 = {0.0f, 0.0f, 20.0f};
	LinearPosition T1 = {0.0f, 0.0f, 20.0f};

	LinearPosition T2 = {0.0f, 0.0f, 20.0f};
	LinearPosition T3 = {0.0f, 0.0f, -20.0f};

	LinearPosition T4 = {0.0f, 0.0f, -20.0f};
	LinearPosition T5 = {0.0f, 0.0f, -20.0f};

	HermiteSegment seg1(P0, P1, T0, T1);
	HermiteSegment seg2(P1, P2, T2, T3);
	HermiteSegment seg3(P2, P3, T4, T5);

	TrajectoryPath path;
	path.addSegment(seg1);
	path.addSegment(seg2);
	path.addSegment(seg3);
	path.build();
	float S = path.getTotalLength();
	SinusoidalSCurveProfile profile(S, V_max, T_acc);
	float Ttotal = profile.getTotalTime();
	float t = 0.0;
	while(t<Ttotal){
		MotionState ms = profile.evaluate(t);
		target_position = path.evaluateByDistance(ms.s);
		IKResult my_solution = inverseKinematics(target_position);
		ok = my_solution.hasSolution;
		target_angle = my_solution.q;
		t+=Ts;
		mantenerPantallaTurnoRobot();
		delay(50);
	}
	MotionState ms = profile.evaluate(Ttotal);
	target_position = path.evaluateByDistance(ms.s);
	Serial.print("final s2: ");
	printSetPointPosition();
}

void returnHome(void){
	LinearPosition P0 = my_robot.p;
	LinearPosition P1 = {my_robot.p.x, my_robot.p.y, z_elevacion};
	LinearPosition P2 = home_position;

	LinearPosition T0 = {0.0f, 0.0f, 20.0f};
	LinearPosition T1 = {0.0f, 0.0f, 20.0f};

	LinearPosition T2 = {0.0f, 0.0f, 20.0f};
	LinearPosition T3 = {0.0f, 0.0f, 20.0f};

	HermiteSegment seg1(P0, P1, T0, T1);
	HermiteSegment seg2(P1, P2, T2, T3);

	TrajectoryPath path;
	path.addSegment(seg1);
	path.addSegment(seg2);
	path.build();
	float S = path.getTotalLength();
	SinusoidalSCurveProfile profile(S, V_max, T_acc);
	float Ttotal = profile.getTotalTime();
	float t = 0.0;
	while(t<Ttotal){
		MotionState ms = profile.evaluate(t);
		target_position = path.evaluateByDistance(ms.s);
		IKResult my_solution = inverseKinematics(target_position);
		ok = my_solution.hasSolution;
		target_angle = my_solution.q;
		t+=Ts;
		mantenerPantallaTurnoRobot();
		delay(50);
	}
	MotionState ms = profile.evaluate(Ttotal);
	target_position = path.evaluateByDistance(ms.s);
	Serial.print("final s3: ");
	printSetPointPosition();
}

void pickAndPlace(LinearPosition P_pick, LinearPosition P_place){
	pick(P_pick);
	delay(1000);
	mantenerPantallaTurnoRobot();
	Serial.print("pick: ");
	printPosition();
    
	closeGripperSmooth();

	place(P_place);
	delay(1000);
	mantenerPantallaTurnoRobot();
	Serial.print("place: ");
	printPosition();
    
	openGripperSmooth();

	returnHome();
	delay(1000);
	mantenerPantallaTurnoRobot();
	Serial.print("return: ");
	printPosition();
}
