/*
# Copyright (C) 2024-2025 Nithin PS.
# This file is part of Pyrebel.
#
# Pyrebel is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, either 
# version 3 of the License, or (at your option) any later version.
#
# Pyrebel is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Pyrebel.
# If not, see <https://www.gnu.org/licenses/>.
#
*/


#include <Arduino_LSM6DSOX.h>

float Ax, Ay, Az;
float Gx, Gy, Gz;

unsigned long last_call=millis();
const int greenLedPin=12;
const int blueLedPin=11;
const int whiteLedPin=10;

const long interval = 250;
int greenLedState=LOW;
int blueLedState=LOW;
int whiteLedState=LOW;
unsigned long greenLedPreviousMillis = 0;  // will store last time LED was updated
unsigned long blueLedPreviousMillis = 0;  // will store last time LED was updated
unsigned long whiteLedPreviousMillis = 0;  // will store last time LED was updated
int flipGreen=HIGH;
int flipBlue=HIGH;
int flipWhite=HIGH;

void on_led_sub(int val){
  if(val==0){
    digitalWrite(LED_BUILTIN,LOW);   // blink the led
  }else if (val==1){
    digitalWrite(LED_BUILTIN,HIGH);
  }
}

void on_led_green_sub(int val){
  if(val==0){
    digitalWrite(greenLedPin,LOW);   // blink the led
    greenLedState=0;
  }else if (val==1){
    digitalWrite(greenLedPin,HIGH);
    greenLedState=1;
  }
}

void on_led_blue_sub(int val){
  if(val==0){
    digitalWrite(blueLedPin,LOW);   // blink the led
    blueLedState=0;
  }else if (val==1){
    digitalWrite(blueLedPin,HIGH);
    blueLedState=1;
  }
}

void on_led_white_sub(int val){
  if(val==0){
    digitalWrite(whiteLedPin,LOW);   // blink the led
    whiteLedState=0;
  }else if (val==1){
    digitalWrite(whiteLedPin,HIGH);
    whiteLedState=1;
  }
}

void on_motor_ctrl_sub1(int speed){
  last_call=millis();
  if (speed>0){
    analogWrite(A1,LOW);
    analogWrite(A0,speed);  
  }else{
    analogWrite(A0,LOW);
    analogWrite(A1,-speed);
  }  
  delay(500);
  stop_zumo();
}

void on_motor_ctrl_sub2(int speed){
  last_call=millis();
  if (speed>0){
    analogWrite(A3,LOW);
    analogWrite(A2,speed);  
  }else{
    analogWrite(A2,LOW);
    analogWrite(A3,-speed);
  }  
  delay(500);
  stop_zumo();
}

void on_zumo_move(int speed){
  last_call=millis();
  if (speed>0){
    analogWrite(A1,LOW);
    analogWrite(A0,speed);
    analogWrite(A3,LOW);
    analogWrite(A2,speed);  
  }else{
    analogWrite(A0,LOW);
    analogWrite(A1,-speed);
    analogWrite(A2,LOW);
    analogWrite(A3,-speed);
  }  
  delay(500);
  stop_zumo();
}

void on_zumo_spin(int speed){
  last_call=millis();
  if (speed>0){
    analogWrite(A1,LOW);
    analogWrite(A0,speed);
    analogWrite(A3,speed);
    analogWrite(A2,LOW);  
  }else{
    analogWrite(A0,LOW);
    analogWrite(A1,-speed);
    analogWrite(A2,-speed);
    analogWrite(A3,LOW);
  }  
  delay(500);
  stop_zumo();
}
void stop_zumo(){
  analogWrite(A0,LOW);
  analogWrite(A1,LOW);
  analogWrite(A2,LOW);
  analogWrite(A3,LOW);
}

void setup()
{ 
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(A0,OUTPUT);
  pinMode(A1,OUTPUT);
  pinMode(A2,OUTPUT);
  pinMode(A3,OUTPUT);
  pinMode(greenLedPin,OUTPUT);
  pinMode(blueLedPin,OUTPUT);
  pinMode(whiteLedPin,OUTPUT);
  if (!IMU.begin()) {
    while (1);
  }
  Serial.begin(115200);
}

void loop()
{ 
  /*
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(Ax, Ay, Az);
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(Gx, Gy, Gz);
  }   
  */
  if(millis()-last_call>2000){
    stop_zumo();    
  }
  if (Serial.available() > 0) {
    String state = Serial.readStringUntil('\n');
    if (state.startsWith("moveforward_")) {
      int speed = state.substring(12).toInt();
      on_zumo_move(speed);
    } else if (state.startsWith("movebackward_")) {
      int speed = state.substring(13).toInt();
      on_zumo_move(-speed);   
    } else if (state.startsWith("spincw_")) {
      int speed = state.substring(7).toInt();
      on_zumo_spin(speed);   
    } else if (state.startsWith("spinacw_")) {
      int speed = state.substring(8).toInt();
      on_zumo_spin(-speed);          
    } else if (state.startsWith("blinkled_")) {
      int status = state.substring(9).toInt();
      on_led_sub(status);         
    } else if (state.startsWith("blinkledgreen_")) {
      int status = state.substring(14).toInt();
      on_led_green_sub(status);          
    } else if (state.startsWith("blinkledblue_")) {
      int status = state.substring(13).toInt();
      on_led_blue_sub(status);          
    } else if (state.startsWith("blinkledwhite_")) {
      int status = state.substring(14).toInt();
      on_led_white_sub(status);          
    }
  }
  
  if (greenLedState==HIGH){
    unsigned long greenLedCurrentMillis = millis();
    if (greenLedCurrentMillis - greenLedPreviousMillis >= interval) {
      // save the last time you blinked the LED
      greenLedPreviousMillis = greenLedCurrentMillis;
      if (flipGreen){
        digitalWrite(greenLedPin,HIGH);
        flipGreen=LOW;
      }else{
        digitalWrite(greenLedPin,LOW);
        flipGreen=HIGH;
      }
    }
  }
  if (blueLedState==HIGH){
    unsigned long blueLedCurrentMillis = millis();
    if (blueLedCurrentMillis - blueLedPreviousMillis >= interval) {
      // save the last time you blinked the LED
      blueLedPreviousMillis = blueLedCurrentMillis;
      if (flipBlue){
        digitalWrite(blueLedPin,HIGH);
        flipBlue=LOW;
      }else{
        digitalWrite(blueLedPin,LOW);
        flipBlue=HIGH;
      }
    }
  }
  if (whiteLedState==HIGH){
    unsigned long whiteLedCurrentMillis = millis();
    if (whiteLedCurrentMillis - whiteLedPreviousMillis >= interval) {
      // save the last time you blinked the LED
      whiteLedPreviousMillis = whiteLedCurrentMillis;
      if (flipWhite){
        digitalWrite(whiteLedPin,HIGH);
        flipWhite=LOW;
      }else{
        digitalWrite(whiteLedPin,LOW);
        flipWhite=HIGH;
      }
    }
  }
  delay(1);
}

