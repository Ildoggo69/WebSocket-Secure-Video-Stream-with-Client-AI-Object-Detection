import cv2
import asyncio
import websockets
import ssl
import logging
import os
from dotenv import load_dotenv

load_dotenv()

IP_REMOTO = os.getenv('IP_REMOTO')
LOG_FILE = os.getenv('LOG_FILE')
CERT_PEM = os.getenv('CERT_PEM')
KEY_PEM = os.getenv('KEY_PEM')


# Configurazione del logging
logfile = {LOG_FILE}
logging.basicConfig(filename=logfile, level=logging.INFO, format="%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s")
logger = logging.getLogger()

# Elenco dei client connessi
connected_clients = set()

# Funzione per gestire i client
async def handle_client(websocket, path):
    # Aggiungi il client all'elenco
    connected_clients.add(websocket)
    logger.info(f"Nuovo client connesso. Totale: {len(connected_clients)}")

    try:
        # Mantieni la connessione aperta
        await websocket.wait_closed()
    finally:
        # Rimuovi il client quando si disconnette
        connected_clients.remove(websocket)
        logger.info(f"Client disconnesso. Totale: {len(connected_clients)}")

# Funzione per trasmettere i frame ai client connessi
async def broadcast_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Errore durante la lettura del frame.")
                continue

            # Codifica il frame come JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            # Invia il frame a tutti i client connessi
            if connected_clients:
                await asyncio.wait([client.send(frame_data) for client in connected_clients])

            # Frame rate ~30 FPS
            await asyncio.sleep(0.150)
    finally:
        cap.release()

# Funzione per avviare il server WSS
async def start_server():
    # Configurazione SSL
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile={CERT_PEM}, keyfile={KEY_PEM})
    
    # Ignora la verifica del certificato
    ssl_context.verify_mode = ssl.CERT_NONE

    # Avvia il server WebSocket sicuro (WSS)
    server = await websockets.serve(handle_client, "0.0.0.0", 8765, ssl=ssl_context)
    logger.info("Server WebSocket sicuro in esecuzione su wss://0.0.0.0:8765")

    # Esegui il server
    await server.wait_closed()

# Funzione principale che avvia sia il server che il broadcast video
async def main():
    await asyncio.gather(start_server(), broadcast_video())
