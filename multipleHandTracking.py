import cv2
from cvzone.HandTrackingModule import HandDetector



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

while True:

    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        first_hand = hands[0]
        first_hand_landmarks = first_hand['lmList']
        first_hand_boundingBox = first_hand['bbox']
        first_hand_centerPoint = first_hand['center']
        first_hand_type = first_hand['type']
        first_hand_fingers = detector.fingersUp(first_hand)

        if len(hands)==2:
            second_hand = hands[1]
            second_hand_landmarks = second_hand['lmList']
            second_hand_boundingBox = second_hand['bbox']
            second_hand_centerPoint = second_hand['center']
            second_hand_type = second_hand['type']
            second_hand_fingers = detector.fingersUp(second_hand)

            #print(first_hand_fingers, second_hand_fingers)
            #length, info, img = detector.findDistance(first_hand_landmarks[8], second_hand_landmarks[8], img)
            #length, info, img = detector.findDistance(first_hand_centerPoint, second_hand_centerPoint, img)



    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break





cap.release()
cv2.destroyAllWindows()
