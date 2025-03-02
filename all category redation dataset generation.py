from ultralytics import YOLO
import cv2
import os
import shutil
import torch

def prepare_dataset(original_dir, blurred_dir, output_dir):
    """ Prepare dataset for YOLO training by extracting blurred regions as labels. """
    os.makedirs(output_dir, exist_ok=True)
    images_output = os.path.join(output_dir, 'images')
    labels_output = os.path.join(output_dir, 'labels')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    
    for filename in os.listdir(original_dir):
        original_path = os.path.join(original_dir, filename)
        blurred_path = os.path.join(blurred_dir, filename)
        
        if not os.path.exists(blurred_path):
            continue
        
        original = cv2.imread(original_path)
        blurred = cv2.imread(blurred_path)
        
        if original is None or blurred is None:
            print(f"Skipping {filename} due to loading error.")
            continue

        diff = cv2.absdiff(original, blurred)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width, _ = original.shape
        label_filename = os.path.join(labels_output, os.path.splitext(filename)[0] + ".txt")
        
        with open(label_filename, 'w') as label_file:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
                w_norm, h_norm = w / width, h / height
                label_file.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
        
        shutil.copy(original_path, os.path.join(images_output, filename))

def train_model(data_yaml, epochs=40, export_model=False):
    """ Train YOLOv8 on the prepared dataset. """
    model = YOLO("yolov8n.pt")  # Load pre-trained model
    model.train(data=data_yaml, epochs=40, imgsz=416, batch=8, device="cuda", workers=4)

    
    if export_model:
        model.export(format="onnx")  # Export model for integration

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Prepare dataset
    prepare_dataset(r"C:\Arun\College\Hackthon\all category dataset",
                    r"C:\Arun\College\Hackthon\all category blur dataset",
                    r"C:\Arun\College\Hackthon\all dataset yolo para")

    # Train YOLO model
    train_model(r"C:\Arun\College\Hackthon\all dataset yolo para\data.yaml", export_model=True)
