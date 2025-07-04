# zumo-jetson

A robot that learns and recognizes patterns on the go.

1. The robot is based on Jetson Nano.

2. The demo script reads the camera frames. Masks the blob, here green (for recognizing) and blue (for learning), using color based segmentation by converting the frames to HSV color space. A check is applied to avoid blobs overlapping with the edge of the frame. The masked image is then inputted to pyrebel which learns/recognizes the blob (using https://github.com/ps-nithin/pyrebel#5-pattern-recognition-demo). Here the recognized symbols are mapped with audio recordings made during learning the pattern. The recorded audio is played back during recognition.

3. The robot uses an Arduino Nano RP2040 Connect as motor controller. It is connected to jetson nano over usb. The arduino firmware has common functions to drive motors and leds connected to it. For example, sending "blinkledwhite_1" over serial to the arduino turns on the white led.

4. Three leds (white, green and blue) connected to the arduino shows the current state of the program. The white led blinks when the program is ready to accept input. The green led blinks when a green blob is detected in the input and program tries to recognize the input. The blue led blinks when a blue blob is detected in the input and the program tries to learn the input. When both the green and the blue led blinks it indicates that both green and blue blobs are present in the input and one may be removed.

5. A speaker and microphone is connected to jetson nano over usb. A Raspberry pi camera module v2 is connected to csi camera port of the jetson nano. Wireless connectivity or internet is not needed to run the program but is used for setting up jetson nano.

# Build
<p align="center"><img src="images/zumo-jetson1.jpg"></img></p><br>

1. Zumo chassis (Pololu - Zumo Chassis Kit (No Motors) https://www.pololu.com/product/1418) act as the main chassis.
2. Motors (Pololu - 298:1 Micro Metal Gearmotor HPCB 6V https://www.pololu.com/product/3069)
3. Motor driver (Pololu - DRV8833 Dual Motor Driver Carrier https://www.pololu.com/product/2130)
4. Voltage regulator (Pololu - 5V Step-Up/Step-Down Voltage Regulator S9V11F5 https://www.pololu.com/product/2836)
5. Arduino Nano RP2040 connect as motor controller
6. Raspberry Pi Camera v2 for camera with acrylic camera mount
7. USB microphone
8. USB speaker
9. AC8265 Wireless NIC for Jetson Nano for wifi
10. Powered by 4xAA batteries.

# Setup
1. Install jetson-utils
https://github.com/dusty-nv/jetson-utils

2. Install pyrebel 
`sudo python3 -m pip install pyrebel`

# Running the script
`git clone https://github.com/ps-nithin/zumo-jetson`<br>
`cd zumo-jetson/scripts`<br>
`sudo python3 pyrebel_main_learn_sound_jetson.py --camera 1`<br>
