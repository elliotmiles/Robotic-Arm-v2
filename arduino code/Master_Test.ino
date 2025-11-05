#include <Wire.h>

void setup() {
  // put your setup code here, to run once:
  Wire.begin();

}

void loop() {
  // put your main code here, to run repeatedly:

  Wire.beginTransmission(8); 
  Wire.write(1);             
  Wire.endTransmission();
  delay(1000);
  
  Wire.beginTransmission(8);
  Wire.write(2);             
  Wire.endTransmission();
  delay(1000); 

  Wire.beginTransmission(8);
  Wire.write(1);             
  Wire.endTransmission();
  delay(1000);   

  Wire.beginTransmission(8);
  Wire.write(3);             
  Wire.endTransmission();
  delay(1000);   
}
