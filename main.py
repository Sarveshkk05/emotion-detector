from model.train import train_model
from model.inference import run_inference

if __name__ == "__main__":
    choice = input("Enter 'train' or 'infer': ")

    if choice == "train":
        train_model()
    else:
        run_inference()