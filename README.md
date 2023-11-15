# Structured_Light_Camera
Central Codebase for Team Light Work


Steps for connecting to Pi via SSH on your computer
1. Ensure Pi is powered on
2. Connect Pi to your computer via ethernet cable (ethernet to USB-C adapter in capstone tupperware)
3. Ensure the Pi is reachable from Pi (use cmd prompt: ping raspberrypi)
4. SSH into Pi using PUTTY
5. Use username: lightwork, password: TEAM10
6. Change directory to scripts and then into Structured_Light_Camera (this will serve as the central codebase)
7. Run command export DISPLAY=:0 (this will tell the Pi to use its own display)


Steps for running Calibration
1. Navigate to ProjectorCameraCalibration director within Structured_Light_Camera
2. For initial image caputure, run the command "libcamerify python3 imageCapture.py". This will project a series of images to the screen and save the images to the calibImages directory.
3. Run the following command: "python3 hammingColorDecode.py"
4. Run the following command: "python3 projectorCali.py
5. Coefficents will outputted to the current directory under the filename: data.npz


Steps for Binary Pattern Image Capture:
1. Navigate to the main directory, Structured_Light_Camera
2. Run the following command "libcamerify python3 GrayCoding.py"
3. Projected images will be shown on the screen and images will stored in the directory grayCodePics
