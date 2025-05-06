# Import modules
import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=1, complexity=1, detectionCon=0.7, trackCon=0.7):
        # Initialize the hand detector with configurable parameters
        self.mode = mode                  # Mode of the detector (static image or not)
        self.maxHands = maxHands          # Maximum number of hands to detect
        self.complexity = complexity      # Complexity of the hand landmark model
        self.detectionCon = detectionCon  # Minimum confidence value for hand detection
        self.trackCon = trackCon          # Minimum confidence value for hand tracking

        # Initialize MediaPipe Hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)

        # Utility for drawing hand landmarks
        self.mpDraw = mp.solutions.drawing_utils

        # List of fingertip landmark IDs
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Convert image to RGB for MediaPipe and process it
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If hand landmarks are detected and drawing is enabled
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Initialize list to store landmarks of the detected hand
        self.lmList = []

        if self.results.multi_hand_landmarks:
            # Get landmarks of the specified hand number
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # Convert landmark position to pixel coordinates
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Add landmark ID and coordinates to the list
                self.lmList.append([id, cx, cy])

                if draw:
                    # Draw a circle at each landmark position
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        # List to store the state (up/down) of each finger
        fingers = []

        if not self.lmList:
            return fingers

        # Check thumb: compare x-coordinates of tip and lower joint
        fingers.append(1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] else 0)

        # Check other four fingers: compare y-coordinates of tip and middle joint
        for i in range(1, 5):
            fingers.append(1 if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2] else 0)

        return fingers
