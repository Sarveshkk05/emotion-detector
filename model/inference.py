import torch
import cv2
from PIL import Image

from model.model import CNN
from config import *
from utils.preprocessing import val_transform

def run_inference():

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = CNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 🔥 Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Cannot access webcam")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔥 Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            tensor = val_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, 1)

            label = class_names[pred.item()]

            # Draw box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # Show label
            cv2.putText(frame, f"{label} {conf.item():.2f}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,255,0), 2)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()