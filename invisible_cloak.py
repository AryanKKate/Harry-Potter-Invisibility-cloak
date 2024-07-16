import cv2
import numpy as np
import time

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(3)

# Capture the background frame
for i in range(60):
    check, backGround = video.read()
    if not check:
        print("Error: Could not read frame from video source.")
        continue

backGround = np.flip(backGround, axis=1)

while video.isOpened():
    check, img = video.read()
    if not check:
        print("Error: Could not read frame from video source.")
        break

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for red color detection in HSV space
    lowerRed1 = np.array([0, 120, 50])
    upperRed1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)

    lowerRed2 = np.array([170, 120, 50])
    upperRed2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)

    mask1 = mask1 + mask2

    # Refining the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(img, img, mask=mask2)
    res2 = cv2.bitwise_and(backGround, backGround, mask=mask1)

    final = cv2.addWeighted(res1, 1, res2, 1, 0)
    cv2.imshow("Invisible Cloak", final)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()