// Include the AccelStepper Library
#include <AccelStepper.h>

const int shoulderDirPin = 4;
const int shoulderStepPin = 5;

const int switchPin = 13;



#define motorInterfaceType 1
AccelStepper shoulderMotor(motorInterfaceType, shoulderStepPin, shoulderDirPin);

void setup() {
  // put your setup code here, to run once:
  pinMode(switchPin, INPUT_PULLUP);
  shoulderMotor.setMaxSpeed(1000);
  shoulderMotor.setAcceleration(50);
}

void loop() {
  // put your main code here, to run repeatedly:

}
