import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
gesture_delay = 0.3  # Reduced delay for faster saving
last_save_time = 0

# Get input from user
print("Options:")
print("1. Letters (A-Z)")
print("2. Numbers (0-9)")
print("3. Space gesture")
while True:
    choice = input("Enter your choice (1-3): ")
    if choice in ['1', '2', '3']:
        break
    print("Invalid choice. Please enter 1, 2, or 3")

if choice == '1':
    while True:
        user_input = input("Enter the letter you want to collect (A-Z): ").upper()
        if user_input.isalpha() and len(user_input) == 1:
            break
        print("Please enter a single letter (A-Z)")
    category = "Letters"
elif choice == '2':
    while True:
        user_input = input("Enter the number you want to collect (0-9): ")
        if user_input.isdigit() and len(user_input) == 1:
            break
        print("Please enter a single number (0-9)")
    category = "Numbers"
else:
    user_input = "SPACE"
    category = "Special"

# Create directory structure
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

category_dir = os.path.join(data_dir, category)
if not os.path.exists(category_dir):
    os.makedirs(category_dir)

char_dir = os.path.join(category_dir, user_input)
if not os.path.exists(char_dir):
    os.makedirs(char_dir)

# Get existing files to determine starting counter
existing_files = [f for f in os.listdir(char_dir) if f.startswith(user_input)]
counter = len(existing_files)

print(f"\nPress 's' to save {user_input} gestures")
print(f"Starting count: {counter}")
print("Press ESC to exit\n")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Warning: Camera frame not read")
            continue

        hands, img = detector.findHands(img)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Boundary checks with safe margins
            safe_margin = 10
            x1 = max(0, x - offset - safe_margin)
            y1 = max(0, y - offset - safe_margin)
            x2 = min(img.shape[1], x + w + offset + safe_margin)
            y2 = min(img.shape[0], y + h + offset + safe_margin)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:
                try:
                    aspectRatio = h / w

                    if aspectRatio > 1:  # Taller than wide
                        new_width = math.ceil(imgSize / aspectRatio)
                        imgResize = cv2.resize(imgCrop, (new_width, imgSize))
                        width_gap = (imgSize - new_width) // 2
                        imgWhite[:, width_gap:width_gap + new_width] = imgResize
                    else:  # Wider than tall
                        new_height = math.ceil(imgSize * aspectRatio)
                        imgResize = cv2.resize(imgCrop, (imgSize, new_height))
                        height_gap = (imgSize - new_height) // 2
                        imgWhite[height_gap:height_gap + new_height, :] = imgResize

                    cv2.imshow("Hand Crop", imgCrop)
                    cv2.imshow("White Background", imgWhite)
                except Exception as e:
                    print(f"Image processing error: {e}")
                    continue

        cv2.imshow(f"Collecting: {user_input}", img)

        key = cv2.waitKey(1)
        current_time = time.time()

        if key == ord("s") and (current_time - last_save_time) > gesture_delay:
            last_save_time = current_time
            counter += 1
            filename = os.path.join(char_dir, f"{user_input}_{counter:04d}.jpg")  # 4-digit zero-padded number

            try:
                cv2.imwrite(filename, imgWhite)
                print(f"\rSaved {user_input} gesture #{counter}", end='', flush=True)
            except Exception as e:
                print(f"\nFailed to save image: {e}")
                counter -= 1
        elif key == 27:  # ESC key to exit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\nCollection complete for '{user_input}'")
    print(f"Total {user_input} gestures saved: {counter}")
    print(f"Location: {char_dir}")