import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained model (ONNX or PyTorch format)
model_path = r"runs\detect\train20\weights\best.onnx"  # Change to best.pt if testing PyTorch model
model = YOLO(model_path)

# Test image path (Change this to your test image)
image_path = r"C:\Arun\College\Hackthon\all category dataset\image_079.jpg"

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image {image_path}")
    exit()

# Copy for original and blurred versions
original_image = image.copy()
blurred_image = image.copy()

# Run inference
results = model.predict(image, imgsz=416, conf=0.25)  # Adjust confidence threshold if needed

# Draw bounding boxes and blur detected regions
for result in results:
    boxes = result.boxes.xyxy  # Get bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert to int
        
        # Draw bounding box on the original image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        
        # Apply Gaussian blur to the detected area
        roi = blurred_image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (25, 25), 30)
        blurred_image[y1:y2, x1:x2] = blurred_roi  # Replace with blurred region

# Save the original detection and blurred detection images
cv2.imwrite("detection_original.jpg", original_image)
cv2.imwrite("detection_blurred.jpg", blurred_image)

print("Original detection saved as detection_original.jpg")
print("Blurred detection saved as detection_blurred.jpg")

# Convert images to RGB for Matplotlib display
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

# Show both images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_rgb)
axes[0].set_title("Original Detection")
axes[0].axis("off")

axes[1].imshow(blurred_rgb)
axes[1].set_title("Blurred Detection")
axes[1].axis("off")

plt.show()
