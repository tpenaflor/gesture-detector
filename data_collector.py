import csv
import os

# Create a CSV file to store landmarks
def create_csv():
    if not os.path.exists('gesture_data.csv'):
        with open('gesture_data.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            # Header row (21 landmarks x/y/z + label)
            header = ['label']
            for i in range(21):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)

# Save landmarks to CSV
def save_landmarks_to_csv(label, landmarks):
    with open('gesture_data.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [label]
        for lm in landmarks.landmark:
            row += [lm.x, lm.y, lm.z]
        writer.writerow(row)