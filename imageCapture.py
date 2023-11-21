mport cv2
import os
import random
import subprocess
import time
import socket 
import signal
HOST = '192.168.40.190'
PORT= 50000


# Folder containing the background images
background_folder = "/home/lightwork/BinaryCodingPictures"


# Set the path to feh
feh_path = "/usr/bin/feh"  # Adjust the path if needed

# Capture an image from the camera
def capture_image(i):
    frameNumber+=1
    print(f"Capturing Blank Image:{i}")
    ret, frame = camera.read()
    if ret:
        image_filename = f'captured_image{i}.jpg'
        print(image_filename)
        succ = cv2.imwrite(image_filename, frame)
        if not succ:
            print(f"failed to save {i}")
        else:
            print(succ)
        return image_filename
    else:
        print(f"Failed to capture image: {i}")
        return None


# Main loop
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
camera.set(cv2.CAP_PROP_EXPOSURE, -50)
test = camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ret, frame = camera.read()
print(test)
os.environ['DISPLAY']=":0.0"
frameNumber = 0
delta=0
current=0
previous=0
iter=0
folder = "testCalib/"
while True:
    current = time.time()
    delta += current - previous
    previous =current
    ret, frame = camera.read()
    if delta >10:
        delta = 0 
        if not ret:
            print("Image failed")
            break
            
        fr = int(frameNumber/3)
        if frameNumber % 3 ==0:
            image_filename = f'{fr}.jpg'
            command = subprocess.Popen("feh --fullscreen -Y white_image.png", shell=True)
            cv2.waitKey(1000)
            for i in range(20):
                ret, frame = camera.read()
            if ret:
                 succ = cv2.imwrite(f"{folder}"+image_filename, frame)
                 if not succ:
                    print("Image did not save")
            else:
                 print("Image Failed")
            subprocess.run(f"kill -9 {command.pid+1}", shell=True)
            command.terminate()
        elif frameNumber % 3 == 1:
            feh_command = "feh --fullscreen pattern_x.png &"
            feh_process=subprocess.Popen(feh_command, shell=True)
            cv2.waitKey(1000)
            image_filename = f'{fr}x.jpg'
            for i in range(20):
                ret, frame = camera.read()
            if ret:
                 succ = cv2.imwrite(f"{folder}"+image_filename, frame)
            subprocess.run(f"kill -9 {feh_process.pid +1}", shell=True)
            feh_process.terminate()
        else:
            feh_command = "feh --fullscreen pattern_y.png &"
            feh_process=subprocess.Popen(feh_command, shell=True)
            cv2.waitKey(1000)
            image_filename = f'{fr}y.jpg'
            for i in range(20):
                ret, frame = camera.read()
            image_filename = f'{fr}y.jpg'
            succ = cv2.imwrite(f"{folder}"+image_filename, frame)
            subprocess.run(f"kill -9 {feh_process.pid +1}", shell=True)
            feh_process.terminate()
        frameNumber += 1
        
    # # Capture an blank image
    # print(camera.isOpened())
    # if frameNumber % 3 ==0:
        # ret, blankFrame = camera.read()
        # if ret:
            # image_filename = f'no_pattern:{frameNumber}.jpg'
            # succ = cv2.imwrite("calibImages/"+image_filename, blankFrame)
            # if not succ:
                # print(succ)
                # exit("Failed to save blank image")
        # else:
            # print(f"Failed to capture blank image: {frameNumber}")
            
        # #time.sleep(1)
    # elif frameNumber % 3 == 1:
        # #Project x pattern image
        # # feh_command = "feh --fullscreen pattern_x.png &"
        # # feh_process=subprocess.Popen(feh_command, shell=True)
        # try:
            # time.sleep(1)
            # print(f"Capturing X Image:{frameNumber}")
            # ret, xFrame = camera.read()
            # if ret:
                # image_filename = f'pattern_x:{frameNumber}.jpg'
                # succ = cv2.imwrite("calibImages/"+image_filename, xFrame)
                # if not succ:
                    # exit("Failed to save X image")
            # else:
                # print(f"Failed to capture X image: {frameNumber}")
        # except:
            # pass
    # else:
        # #Project y pattern image
        # #feh_command = "feh --fullscreen pattern_y.png &"
        # #feh_process=subprocess.Popen(feh_command, shell=True)
        # try:
            # time.sleep(1)
            # print(f"Capturing Y Image:{frameNumber}")
            # ret, yFrame = camera.read()
            # if ret:
                # image_filename = f'pattern_y:{frameNumber}.jpg'
                # succ = cv2.imwrite("calibImages/"+image_filename, yFrame)
                # if not succ:
                    # exit("Failed to save y Image")
                # print(feh_process.pid)
            # else:
                # print(f"Failed to capture y image: {frameNumber}")
            # #subprocess.run(f"kill -9 {feh_process.pid +1}", shell=True)
            # #feh_process.terminate()
        # #except subprocess.CalledProcessError as e:
            # #print(f"Error running feh {e}")
        # #time.sleep(1)
        # except:
            # pass
        
    if frameNumber > 32: 
        break
    
    
    
    
    
# Note: You can stop the script by pressing Ctrl+C in the terminal.
camera.release()
cv2.destroyAllWindows()
