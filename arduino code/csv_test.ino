void setup() {
  Serial.begin(115200);
  Serial.println("Connection established");
  Serial.println("Homing motors...");

  //begin homing sequence

}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    Serial.println("Received joint angles");


    // Split by commas
    int firstComma = msg.indexOf(',');
    int secondComma = msg.indexOf(',', firstComma + 1);
    int thirdComma = msg.indexOf(',', secondComma + 1);

    String field1 = msg.substring(0, firstComma);
    String field2 = msg.substring(firstComma + 1, secondComma);
    String field3 = msg.substring(secondComma + 1, thirdComma);
    String field4 = msg.substring(thirdComma + 1);

    float a = field1.toFloat();
    float b = field2.toFloat();
    float c = field3.toFloat();
    float d = field4.toFloat();

    Serial.print("A: ");
    Serial.println(a);
    Serial.print("B: ");
    Serial.println(b);
    Serial.print("C: ");
    Serial.println(c);
    Serial.print("D: ");
    Serial.println(d);

  }

}
