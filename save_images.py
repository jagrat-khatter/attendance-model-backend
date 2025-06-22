# save_images.py

import cv2
import os

person_name = "Jagrat"  # <-- Change to desired name
save_dir = os.path.join("dataset", person_name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):
        path = os.path.join(save_dir, f"img{count}.jpg")
        cv2.imwrite(path, frame)
        print(f"[+] Saved: {path}")
        count += 1

    elif key & 0xFF == ord('q') or count >= 10:
        break

cap.release()
cv2.destroyAllWindows()
