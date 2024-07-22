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
out1 = 17  # Left wheel
out2 = 18  # Left wheel
out3 = 27  # Right wheel
out4 = 22  # Right wheel

# GPIO setup for L298N motor driver
GPIO.setmode(GPIO.BCM)
GPIO.setup(out1, GPIO.OUT)
GPIO.setup(out2, GPIO.OUT)
GPIO.setup(out3, GPIO.OUT)
GPIO.setup(out4, GPIO.OUT)

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
                        # Adjust this value based on your camera's calibration
                        focal_length = 1000
                        distance = (focal_length * 50) / (box[2] + box[3])

                        if distance < 60:
                            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
                            move_right_forward()
                            move_left_backward()
                            time.sleep(1)
                            move_both_forward()
                            time.sleep(1.5)
                            move_right_backward()
                            move_left_forward()
                            time.sleep(1)
                            move_both_forward()
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
            cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print("Cleaning up GPIO...")
        GPIO.cleanup()
        print("GPIO cleanup complete. Exiting the script.")
