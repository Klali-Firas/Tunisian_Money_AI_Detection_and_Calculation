# Import required packages
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load label map from file
with open("labelmap.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove '???' if it's the first label
if labels[0] == '???':
    labels.pop(0)

# Function to preprocess the image frame
def preprocess_frame(frame):
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model's input size
    resized_frame = cv2.resize(frame_rgb, (input_shape[2], input_shape[1]))
    # Expand dimensions and convert to float32 for compatibility
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)
    input_data /= 255.0  # Normalize to [0, 1] range
    return input_data

# Start video capture loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and set tensor
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract the output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])  # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])  # Confidence scores

    # Iterate through detected objects
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin = int(max(1, boxes[i][0] * frame.shape[0]))
            xmin = int(max(1, boxes[i][1] * frame.shape[1]))
            ymax = int(min(frame.shape[0], boxes[i][2] * frame.shape[0]))
            xmax = int(min(frame.shape[1], boxes[i][3] * frame.shape[1]))

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            label = f"{labels[int(classes[i][0])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame with detection boxes
    cv2.imshow('Object Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
