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

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        tensor = val_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_names[pred.item()]

        cv2.putText(frame, f"{label} {conf.item():.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()