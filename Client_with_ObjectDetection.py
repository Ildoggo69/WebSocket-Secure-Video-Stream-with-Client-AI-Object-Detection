import cv2
import torch
import numpy as np
import os
import websockets
import asyncio
import ssl
from ultralytics import YOLO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables from .env file
load_dotenv()

# Get the remote server IP from environment variables
YOUR_IP = os.getenv('IP_REMOTO')
PATH_TO_CERT = os.getenv('CERT_PEM')
PATH_TO_KEY = os.getenv('KEY_PEM')

# Create the device object for GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Debugging information about the device
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Active device: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Debugging line to verify which device is being used
print(f"Using device: {device}")

# Get the current directory
current_dir = os.path.dirname(__file__)
# Path to the model file
model_path = os.path.join(current_dir, "best.pt")
print(f"Looking for the model file at: {model_path}")

# Check if the model file exists
if not os.path.exists(model_path):
    print("The file does not exist.")
    exit(1)

print("The file exists. Proceeding to load the model.")
try:
    # Load the model on GPU or CPU
    model = YOLO(model_path).to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Executor for parallel processing
executor = ThreadPoolExecutor(max_workers=2)

# Function to perform object detection on a frame
def detect_objects(frame):
    # Disable verbose mode during prediction
    predictions = model.predict(source=frame, save=False, save_txt=False, verbose=False)  
    frame_with_boxes = predictions[0].plot()
    return frame_with_boxes

# Function to draw bounding boxes on the frame
def draw_boxes(frame, results):
    annotated_frame = results[0].plot()
    return annotated_frame

# Function to connect to the WebSocket server
async def connect_to_websocket():
    websocket_url = f"wss://{YOUR_IP}:443"
    print(f"Connecting to WebSocket server: {websocket_url}")
    
    # Paths to the certificate and private key
    cert_path = os.path.join(PATH_TO_CERT, "certificate.pem")
    key_path = os.path.join(PATH_TO_KEY, "private_key.pem")
    
    # Check if the certificate and private key exist
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        print("Certificate or private key not found.")
        exit(1)

    # Set up the SSL context for a secure WebSocket connection
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ssl_context.check_hostname = False
    ssl_context.load_verify_locations(cert_path)

    async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
        print("WebSocket connection established.")
        frame_batch = []
        batch_size = 10
        prev_time = time.time()  # Initial time
        frame_count = 0  # Frame counter

        while True:
            try:
                # Receive frame data from WebSocket
                frame_data = await websocket.recv()
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Error decoding the frame.")
                    continue

                # Resize the frame to 640x480
                frame = cv2.resize(frame, (640, 480))

                frame_batch.append(frame)
                if len(frame_batch) >= batch_size:
                    # Process only the first frame in the batch
                    future = executor.submit(detect_objects, frame_batch[0])
                    frame_with_boxes = future.result()
                    frame_batch = []  # Reset the batch

                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    elapsed_time = current_time - prev_time
                    if elapsed_time >= 1.0:  # Every second
                        fps = frame_count / elapsed_time
                        prev_time = current_time
                        frame_count = 0

                        # Display FPS on the frame
                        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Show the frame with object detection
                    cv2.imshow("Video Stream with Object Detection", frame_with_boxes)

                # Check if the user pressed 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Error in connection or receiving frames: {e}")
                break

        # Destroy OpenCV windows
        cv2.destroyAllWindows()

# Main entry point
if __name__ == "__main__":
    asyncio.run(connect_to_websocket())
