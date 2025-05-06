# Import modules
import cv2
import numpy as np
import os
import hand_tracking_module as htm
import matplotlib.pyplot as plt
from load_data import get_data
from keras.models import load_model

# Define the data directory path
data_dir = r'C:\Users\Noor\PycharmProjects\Categories'

# Function to load the best trained model from a given file path
def load_trained_model(model_path):
    return load_model(model_path)

# Function to preprocess the canvas image for neural network prediction
def preprocess_image(image, target_size=(28, 28)):
    # Convert image to grayscale to focus on structural features
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to highlight the main features in the image
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate the image to close small holes or gaps in the features
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours to identify the main object in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour which likely represents the main object
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the image around this main object
        cropped_image = thresh[y:y+h, x:x+w]

        # Calculate the aspect ratio and resize the image to maintain it
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = target_size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect_ratio)
        scaled_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a blank canvas and place the scaled image in the center
        new_canvas = np.zeros(target_size, dtype=np.uint8)
        x_center = (target_size[0] - new_w) // 2
        y_center = (target_size[1] - new_h) // 2
        new_canvas[y_center:y_center+new_h, x_center:x_center+new_w] = scaled_image
    else:
        # If no contours are found, directly apply adaptive thresholding and resize
        new_canvas = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        new_canvas = cv2.resize(new_canvas, target_size, interpolation=cv2.INTER_AREA)

    # Normalize the canvas for prediction
    image_normalized = new_canvas / 255.0
    image_reshaped = image_normalized.reshape(1, 28, 28, 1)
    return image_reshaped

# Function to predict the category of the doodle drawn on the canvas
def predict_doodle(model, image):
    predictions = model.predict(image)
    return np.argmax(predictions), predictions[0]

# Function to display the prediction and scores for each category
def display_prediction_and_scores(prediction, scores, categories):
    print(f"Predicted Category: {categories[prediction]}")
    for i, score in enumerate(scores):
        print(f"{categories[i]}: {score:.2%}")

# Function to check if the canvas is empty (no drawings)
def is_canvas_empty(canvas):
    return np.all(canvas == 0)

# Load the trained model
best_model_path = r'C:\Users\Noor\PycharmProjects\CSFinalProject\best_model.h5'
model = load_trained_model(best_model_path)

# Define categories for classification
categories = ['bird', 'cat', 'circle', 'house', 'square', 'triangle']

# Function to scale brush thickness based on the current resolution
def get_scaled_thickness(current_width, current_height, original_width=640, original_height=480, original_brush_thickness=10, original_eraser_thickness=40):
    scale_factor = ((current_width / original_width) + (current_height / original_height)) / 2
    scaled_brush_thickness = int(original_brush_thickness * scale_factor)
    scaled_eraser_thickness = int(original_eraser_thickness * scale_factor)
    return scaled_brush_thickness, scaled_eraser_thickness

# Function to scale the size of UI elements based on the current resolution
def get_scaled_size(current_width, current_height, original_width=640, original_height=480, original_rect_size=(20, 20), original_circle_radius=10):
    scale_factor = ((current_width / original_width) + (current_height / original_height)) / 2
    scaled_rect_width = int(original_rect_size[0] * scale_factor)
    scaled_rect_height = int(original_rect_size[1] * scale_factor)
    scaled_circle_radius = int(original_circle_radius * scale_factor)
    return (scaled_rect_width, scaled_rect_height), scaled_circle_radius

# Function to scale the positions of overlay elements based on the current resolution
def scale_overlay_ranges(current_width, current_height, original_width=640, original_height=480, original_overlay_ranges=[(210, 270), (305, 365), (400, 460), (495, 585)]):
    width_scale = current_width / original_width
    return [(int(start_x * width_scale), int(end_x * width_scale)) for (start_x, end_x) in original_overlay_ranges]

# Function to compare user drawing with training data
def display_comparison(user_drawing_path, training_data_path, categories):
    # Load a random sample from the training data
    X, Y = get_data(categories, training_data_path, augment=False)
    sample_image = X[np.random.randint(len(X))].reshape(28, 28) / 255.0

    # Load and preprocess the user's drawing
    user_drawing = cv2.imread(user_drawing_path, cv2.IMREAD_GRAYSCALE)
    user_drawing_preprocessed = preprocess_image(user_drawing).reshape(28, 28)

    # Display both images for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title('Sample from Training Data')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(user_drawing_preprocessed, cmap='gray')
    plt.title('User Drawing (Preprocessed)')
    plt.axis('off')
    plt.show()

# Load header images for the user interface
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Set default drawing settings (starting with the eraser)
header = overlayList[3]  # Use the fourth image in the overlay list as the default header (eraser)
drawColor = (0, 0, 0)    # Default color set to black (for eraser)

# Initialize video capture
cap = cv2.VideoCapture(0)
success, img = cap.read()
height, width, channels = img.shape

# Adjust brush and eraser thickness, and overlay ranges based on the current resolution
brushThickness, eraserThickness = get_scaled_thickness(width, height)
rect_size, circle_radius = get_scaled_size(width, height)
overlay_ranges = scale_overlay_ranges(width, height)

# Create an empty canvas with the same resolution as the webcam
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Initialize the hand tracking module
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0  # Initialize previous x, y coordinates for drawing

# Main loop for the virtual painter application
while True:
    # Read an image frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image for a mirror-like effect

    # Find hands and their landmarks in the image
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get coordinates of the index and middle fingertips
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Determine which fingers are up
        fingers = detector.fingersUp()

        # Tool selection logic based on finger positions
        if fingers[1] and fingers[2]:  # Both index and middle fingers are up
            xp, yp = 0, 0  # Reset previous position
            if y1 < 125:  # Check if fingers are in the header area
                overlay_colors = [(94, 23, 235), (0, 191, 99), (255, 49, 49), (0, 0, 0)]
                for i, (start_x, end_x) in enumerate(overlay_ranges):
                    if start_x <= x1 <= end_x:
                        header = overlayList[i]
                        drawColor = overlay_colors[i]  # Change draw color based on the selected tool
                        break

                # Draw a rectangle on the image to show the selected color/tool
                cv2.rectangle(img, (x1 - rect_size[0], y1 - rect_size[1]),
                              (x2 + rect_size[0], y2 + rect_size[1]), drawColor, cv2.FILLED)

        # Drawing logic when the index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), circle_radius, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Choose brush width based on whether the eraser or a color is selected
            brushWidth = eraserThickness if drawColor == (0, 0, 0) else brushThickness

            # Draw a line on both the image and the canvas
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushWidth)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushWidth)
            xp, yp = x1, y1  # Update the previous position

    # Create an inverted mask and combine it with the canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Display the header and the canvas
    img[0:125, 0:width] = cv2.resize(header, (width, 125))
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    # Key bindings for various functionalities
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Predict the drawing when 'p' is pressed
        if is_canvas_empty(imgCanvas):
            print("There is nothing drawn to predict!")
        else:
            cv2.imwrite('sample.png', imgCanvas)
            preprocessed_img = preprocess_image(imgCanvas)
            prediction, scores = predict_doodle(model, preprocessed_img)
            display_prediction_and_scores(prediction, scores, categories)
            display_comparison('sample.png', data_dir, categories)

    if key == ord('c'):  # Clear the canvas when 'c' is pressed
        imgCanvas = np.zeros((height, width, 3), np.uint8)

    if key == ord('q'):  # Quit the application when 'q' is pressed
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
