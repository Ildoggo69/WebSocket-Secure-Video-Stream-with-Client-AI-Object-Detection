import cv2
import torch
import numpy as np
import os
import websockets
import asyncio
import ssl
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

YOUR_IP = os.getenv('IP_REMOTO')

# Define the execution device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the path of the current directory (where the script is located)
current_dir = os.path.dirname(__file__)

# Specify the relative path to the YOLO model file
model_path = os.path.join(current_dir, "best.pt")

print(f"Looking for the model file at: {model_path}")

# Load the YOLOv8 model
if not os.path.exists(model_path):
    print("The file does not exist.")
    exit(1)

print("The file exists. Proceeding to load the model.")
try:
    # Initialize the YOLO model with custom weights
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Preprocessing function for the model (not required with YOLOv8 built-in preprocessing)
def preprocess(frame):
    return frame  # YOLOv8 handles preprocessing internally

# Function to perform object detection
def detect_objects(frame):
    results = model.predict(source=frame, save=False, save_txt=False)
    return results

# Function to draw bounding boxes on the frame
def draw_boxes(frame, results):
    # Draw the results directly on the frame
    annotated_frame = results[0].plot()
    return annotated_frame

# Function to connect to the WebSocket and receive frames
async def connect_to_websocket():
    websocket_url = f"wss://{YOUR_IP}:443"  # Secure WebSocket URL
    print(f"Connecting to WebSocket server: {websocket_url}")

    # Create an SSL context that ignores invalid certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False  # Disable hostname verification
    ssl_context.verify_mode = ssl.CERT_NONE  # Do not verify the certificate

    # Connect to the WebSocket using the modified SSL context
    async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
        print("WebSocket connection established.")
        
        while True:
            try:
                # Receive frame data from the WebSocket
                frame_data = await websocket.recv()

                # Convert the received data into an image
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Error decoding the frame.")
                    continue

                # Perform object detection
                predictions = detect_objects(frame)

                # Draw bounding boxes on the frame
                frame_with_boxes = draw_boxes(frame, predictions)

                # Display the frame with bounding boxes
                cv2.imshow("Video Stream with Object Detection", frame_with_boxes)

                # Exit the loop by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error in connection or receiving frames: {e}")
                break

        cv2.destroyAllWindows()

# Run the WebSocket connection
if __name__ == "__main__":
    asyncio.run(connect_to_websocket())
