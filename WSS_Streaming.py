import cv2
import asyncio
import websockets
import ssl
import logging
import os
from dotenv import load_dotenv

load_dotenv()

PATH_TO_LOGFILE = os.getenv('LOG_FILE')
PATH_TO_CERT = os.getenv('CERT_PEM')
PATH_TO_KEY = os.getenv('KEY_PEM')


# Configure logging
logfile = PATH_TO_LOGFILE
logging.basicConfig(filename=logfile, level=logging.INFO, format="%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s")
logger = logging.getLogger()

# List of connected clients
connected_clients = set()

# Function to handle connected clients
async def handle_client(websocket, path):
    # Add the client to the set of connected clients
    connected_clients.add(websocket)
    logger.info(f"New client connected. Total: {len(connected_clients)}")

    try:
        # Keep the connection open
        await websocket.wait_closed()
    finally:
        # Remove the client when it disconnects
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(connected_clients)}")

# Function to broadcast video frames to connected clients
async def broadcast_video():
    cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video width to 640 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video height to 480 pixels

    try:
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                logger.error("Error reading frame.")
                continue

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            # Send the frame to all connected clients
            if connected_clients:
                await asyncio.wait([client.send(frame_data) for client in connected_clients])

            # Limit the frame rate to approximately 30 FPS
            await asyncio.sleep(0.150)
    finally:
        cap.release()  # Release the camera when done

# Function to start the WSS server
async def start_server():
    # Configure SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=PATH_TO_CERT, keyfile=PATH_TO_KEY)
    
    # Ignore certificate verification
    ssl_context.verify_mode = ssl.CERT_NONE

    # Start the secure WebSocket server (WSS)
    server = await websockets.serve(handle_client, "0.0.0.0", 8765, ssl=ssl_context)
    logger.info("Secure WebSocket server running on wss://0.0.0.0:8765")

    # Keep the server running
    await server.wait_closed()

# Main function to start both the server and video broadcast
async def main():
    await asyncio.gather(start_server(), broadcast_video())
