import cv2
import torch
from yolov7_utils import load_model, preprocess_frame, run_inference
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv7 model
model = load_model('/home/tie201/road_quality_project/models/best.pt', device)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Set the desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    start_time = time.time()  # Start timer
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the frame
    img_tensor = preprocess_frame(frame)

    # Run inference
    frame = run_inference(model, img_tensor, frame, device)

    # Calculate and display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('YOLOv7 Detection', frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) == ord('q'):
        break
    print("FPS",fps)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
