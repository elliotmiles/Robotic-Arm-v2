#include <AccelStepper.h>

// Base
#define A_DirPin 4
#define A_StepPin 5
#define B_DirPin 6
#define B_StepPin 7
#define C_DirPin 8
#define C_StepPin 9
#define D_DirPin 10
#define D_StepPin 11

//limit swtich pins
#define limit_A A0
#define limit_B A1
#define limit_C A2
#define limit_D A3

#define motorInterfaceType 1

AccelStepper motor_A(motorInterfaceType, A_StepPin, A_DirPin);
AccelStepper motor_B(motorInterfaceType, B_StepPin, B_DirPin);
AccelStepper motor_C(motorInterfaceType, C_StepPin, C_DirPin);
AccelStepper motor_D(motorInterfaceType, D_StepPin, D_DirPin);

long quarterTurn = 0.25 * 200 * (70/20) * 16;
long halfTurn = 0.5 * 200 * (70/20) * 16;
long reverseDist = 200;

void setup() {

  // Analogue pins being used as digital inputs for the homing switches
  pinMode(limit_A, INPUT_PULLUP); 
  pinMode(limit_B, INPUT_PULLUP); 
  pinMode(limit_C, INPUT_PULLUP);  
  pinMode(limit_D, INPUT_PULLUP); 

    //----- A HOMING -----
  motor_A.setMaxSpeed(1000);
  motor_A.setAcceleration(200);

  //fast approach
  motor_A.setSpeed(200);
  while (digitalRead(limit_A) == LOW) {
    motor_A.runSpeed();
  }

  delay(200);

  //reverse
  motor_A.setMaxSpeed(200);
  motor_A.setAcceleration(200);
  motor_A.move(-reverseDist);
  while (motor_A.isRunning()) {
    motor_A.run();
  }


  //slow approach
  motor_A.setSpeed(100);
  while (digitalRead(limit_A) == LOW) {
    motor_A.runSpeed();
  }
  motor_A.setCurrentPosition(0);

  motor_A.moveTo(-quarterTurn); 
  while (motor_A.distanceToGo() != 0) {
    motor_A.run();
  }


    //----- D HOMING -----
  motor_D.setMaxSpeed(1000);
  motor_D.setAcceleration(200);

  //fast approach
  motor_D.setSpeed(200);
  while (digitalRead(limit_D) == LOW) {
    motor_D.runSpeed();
  }

  delay(200);

  //reverse
  motor_D.setMaxSpeed(200);
  motor_D.setAcceleration(200);
  motor_D.move(-reverseDist);
  while (motor_D.isRunning()) {
    motor_D.run();
  }


  //slow approach
  motor_D.setSpeed(100);
  while (digitalRead(limit_D) == LOW) {
    motor_D.runSpeed();
  }
  motor_D.setCurrentPosition(0);

  motor_D.moveTo(-quarterTurn); 
  while (motor_D.distanceToGo() != 0) {
    motor_D.run();
  }

    //----- C HOMING -----
  motor_C.setMaxSpeed(1000);
  motor_C.setAcceleration(200);

  //fast approach
  motor_C.setSpeed(-200);
  while (digitalRead(limit_C) == LOW) {
    motor_C.runSpeed();
  }

  delay(200);

  //reverse
  motor_C.setMaxSpeed(200);
  motor_C.setAcceleration(200);
  motor_C.move(reverseDist);
  while (motor_C.isRunning()) {
    motor_C.run();
  }


  //slow approach
  motor_C.setSpeed(-100);
  while (digitalRead(limit_C) == LOW) {
    motor_C.runSpeed();
  }
  motor_C.setCurrentPosition(0);

    //----- B HOMING -----
  motor_B.setMaxSpeed(1000);
  motor_B.setAcceleration(200);

  //fast approach
  motor_B.setSpeed(-200);
  while (digitalRead(limit_B) == LOW) {
    motor_B.runSpeed();
  }

  delay(200);

  //reverse
  motor_B.setMaxSpeed(200);
  motor_B.setAcceleration(200);
  motor_B.move(reverseDist);
  while (motor_B.isRunning()) {
    motor_B.run();
  }


  //slow approach
  motor_B.setSpeed(-100);
  while (digitalRead(limit_B) == LOW) {
    motor_B.runSpeed();
  }
  motor_B.setCurrentPosition(0);  

}  

void loop() {

}
