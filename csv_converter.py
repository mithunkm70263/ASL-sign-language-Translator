from csv import writer
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True,
            max_num_hands = 1 )

DATASET_PATH = "/Users/mithunkm/Desktop/Sign recognition IDT/archive/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "landmarks.csv"

with open(OUTPUT_CSV,"w",newline="") as file:
    writer = csv.writer(file)
    header = []
    # 21 landmarks × (x,y,z)
    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")

    # Add label column
    header.append("label")

    # Write header row
    writer.writerow(header)

    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH,label)
        # Skip non-folder files

        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")
        for image_name in os.listdir(label_path):

            image_path = os.path.join(label_path, image_name)

            # Read image
            image = cv2.imread(image_path)

            # Skip broken images
            if image is None:
                continue

            # Convert BGR → RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe
            results = hands.process(rgb_image)

            # IF HAND DETECTED

            if results.multi_hand_landmarks:

                # Take first hand
                hand_landmarks = results.multi_hand_landmarks[0]

                row = []

                # Extract x,y,z from all 21 landmarks
                for landmark in hand_landmarks.landmark:

                    row.append(landmark.x)
                    row.append(landmark.y)
                    row.append(landmark.z)

                # Add label at end
                row.append(label)

                # Write to CSV
                writer.writerow(row)

# Cleanup
hands.close()

print("Done! landmarks.csv created.")
