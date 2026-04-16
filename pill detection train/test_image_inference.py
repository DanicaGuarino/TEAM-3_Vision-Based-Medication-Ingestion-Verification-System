from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/pill_detector.pt")

# Path to your test image
image_path = "test_images/pill.jpg"

# Run inference
results = model(image_path, conf=0.3)

# Get annotated image
annotated = results[0].plot()

# Resize image for display
display = cv2.resize(annotated, (800, 600))

cv2.imshow("Pill Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.jpg", annotated)

# Show result
cv2.imshow("Pill Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("output.jpg", annotated)

print("Detection complete. Saved to output.jpg")