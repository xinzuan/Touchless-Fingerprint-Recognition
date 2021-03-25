import cv2
import sys

#print("Before cv2.VideoCapture(0)"
#print(cap.grab()
cap = cv2.VideoCapture(0)


print("After cv2.VideoCapture(0): cap.grab() --> " + str(cap.grab()) + "\n")

while True:
    ret, frame = cap.read()
   
    if frame is None:
        print("BYE")
        break

    cv2.imshow('frame', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.destroyWindow('frame')
        break

print("After breaking, but before cap.release(): cap.grab() --> " + str(cap.grab()) + "\n")

cap.release()

print("After breaking, and after cap.release(): cap.grab() --> " + str(cap.grab()) + "\n")
cv2.destroyWindow('frame')
print("After destroy --> " + str(cap.grab()) + "\n")