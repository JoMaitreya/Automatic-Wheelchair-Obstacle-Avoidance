import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Thresholds and other parameters
classNames = []
classFile = "/home/admin/Object_Detection_Files/coco.names"
configPath = "/home/admin/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/admin/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# GPIO pin definitions for L298N motor driver
out1 = 17  # Left wheel forward
out2 = 18  # Left wheel backward
out3 = 27  # Right wheel forward
out4 = 22  # Right wheel backward

# GPIO pin definitions for sensors and push button
TRIG_LEFT = 5
ECHO_LEFT = 6
TRIG_RIGHT = 13
ECHO_RIGHT = 19
IR_LEFT = 24
IR_RIGHT = 25
BUTTON_PIN = 4

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(out1, GPIO.OUT)
GPIO.setup(out2, GPIO.OUT)
GPIO.setup(out3, GPIO.OUT)
GPIO.setup(out4, GPIO.OUT)
GPIO.setup(TRIG_LEFT, GPIO.OUT)
GPIO.setup(ECHO_LEFT, GPIO.IN)
GPIO.setup(TRIG_RIGHT, GPIO.OUT)
GPIO.setup(ECHO_RIGHT, GPIO.IN)
GPIO.setup(IR_LEFT, GPIO.IN)
GPIO.setup(IR_RIGHT, GPIO.IN)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Load the class names
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Functions to control the wheels
def move_right_forward():
    GPIO.output(out3, GPIO.HIGH)
    GPIO.output(out4, GPIO.LOW)

def move_left_backward():
    GPIO.output(out1, GPIO.LOW)
    GPIO.output(out2, GPIO.HIGH)

def move_both_forward():
    GPIO.output(out1, GPIO.HIGH)
    GPIO.output(out2, GPIO.LOW)
    GPIO.output(out3, GPIO.HIGH)
    GPIO.output(out4, GPIO.LOW)

def move_right_backward():
    GPIO.output(out3, GPIO.LOW)
    GPIO.output(out4, GPIO.HIGH)

def move_left_forward():
    GPIO.output(out1, GPIO.HIGH)
    GPIO.output(out2, GPIO.LOW)

def stop_moving():
    GPIO.output(out1, GPIO.LOW)
    GPIO.output(out2, GPIO.LOW)
    GPIO.output(out3, GPIO.LOW)
    GPIO.output(out4, GPIO.LOW)

def read_distance(trig_pin, echo_pin):
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

def getObjects(img, thres, nms, draw=True, objects=[]):
    if img is not None:
        classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
        objectInfo = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        focal_length = 1000  # Use a calibrated focal length
                        distance = (focal_length * 50) / (box[2] + box[3])

                        if distance < 60:
                            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
                            
                            # Determine object position
                            center_x = box[0] + box[2] // 2
                            frame_width = img.shape[1]
                            left_side = center_x < frame_width // 2
                            
                            if left_side:
                                # Object is on the left side, turn right
                                move_right_backward()
                                move_left_forward()
                            else:
                                # Object is on the right side, turn left
                                move_left_backward()
                                move_right_forward()
                                
                            time.sleep(1)  # Adjust the duration of turning
                            move_both_forward()  # Continue moving forward
                        else:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f"Distance: {round(distance, 2)} cm", (box[0] + 10, box[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        return img, objectInfo
    else:
        return None, None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    start_time = time.time()

    try:
        print("Waiting for 10 seconds before starting the wheelchair...")
        while True:
            current_time = time.time()
            if current_time - start_time >= 10:
                move_both_forward()
                print("Wheelchair started moving forward.")
                break

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to read frame")
                continue

            result, objectInfo = getObjects(img, 0.5, 0.2, objects=['person'])
            if result is not None:
                cv2.imshow("Output", result)
            
            # Check push button status
            if not GPIO.input(BUTTON_PIN):
                print("Push button pressed. Stopping...")
                stop_moving()
                break

            cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print("Cleaning up GPIO...")
        GPIO.cleanup()
        print("GPIO cleanup complete. Exiting the script.")
