# zumo-jetson

A robot that learns and recognizes patterns on the go.

1. The robot is based on Jetson Nano.

2. The demo script reads the camera frames. Masks the blob, here green (for recognizing) and blue (for learning), using color based segmentation by converting the frames to HSV color space. A check is applied to avoid blobs overlapping with the edge of the frame. The masked image is then inputted to pyrebel which learns/recognizes the blob. Here the recognized symbols are mapped with audio recordings made during learning the pattern. The recorded audio is played back during recognition.

3. The robot uses an Arduino Nano RP2040 Connect as motor controller. It is connected to jetson nano over usb. The arduino firmware has common functions to drive motors and leds connected to it. For example, sending "blinkledwhite_1" over serial to the arduino turns on the white led.

4. Three leds (white, green and blue) connected to the arduino shows the current state of the program. The white led blinks when the program is ready to accept input. The green led blinks when a green blob is detected in the input and program tries to recognize the input. The blue led blinks when a blue blob is detected in the input and the program tries to learn the input. When both the green and the blue led blinks it indicates that both green and blue blobs are present in the input and one may be removed.

5. A speaker and microphone is connected to jetson nano over usb. A Raspberry pi camera module v2 is connected to csi camera port of the jetson nano. A wireless connectivity nor internet is needed to run the program but can be used for setting up jetson nano.

