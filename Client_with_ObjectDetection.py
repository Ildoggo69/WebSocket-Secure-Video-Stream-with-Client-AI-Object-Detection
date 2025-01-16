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

IP_REMOTO = os.getenv('IP_REMOTO')

# Dispositivo di esecuzione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ottieni il percorso della directory corrente (dove si trova il codice)
current_dir = os.path.dirname(__file__)

# Specifica il percorso relativo del modello
model_path = os.path.join(current_dir, "best.pt")

print(f"Cercando il file del modello in: {model_path}")

# Carica il modello YOLOv8
if not os.path.exists(model_path):
    print("Il file non esiste.")
    exit(1)

print("Il file esiste. Procedo con il caricamento del modello.")
try:
    # Inizializza il modello con i pesi custom
    model = YOLO(model_path)
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    exit(1)

# Funzione di pre-processamento per il modello (non necessaria con YOLOv8 integrato)
def preprocess(frame):
    return frame  # YOLOv8 gestisce il pre-processamento internamente

# Funzione per eseguire il rilevamento degli oggetti
def detect_objects(frame):
    results = model.predict(source=frame, save=False, save_txt=False)
    return results

# Funzione per disegnare le bounding box sul frame
def draw_boxes(frame, results):
    # Disegna i risultati direttamente sul frame
    annotated_frame = results[0].plot()
    return annotated_frame

# Funzione per connettersi al WebSocket e ricevere i frame
async def connect_to_websocket():
    websocket_url = f"wss://{IP_REMOTO}:443"  # URL del WebSocket sicuro
    print(f"Connessione al server WebSocket: {websocket_url}")

    # Crea un contesto SSL che ignora i certificati non validi
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False  # Disabilita il controllo del nome host
    ssl_context.verify_mode = ssl.CERT_NONE  # Non verificare il certificato

    # Connettiamo al WebSocket con il contesto SSL modificato
    async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
        print("Connessione WebSocket stabilita.")
        
        while True:
            try:
                # Ricevi i dati del frame dal WebSocket
                frame_data = await websocket.recv()

                # Converti i dati ricevuti in un'immagine
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Errore nel decodificare il frame.")
                    continue

                # Esegui il rilevamento degli oggetti
                predictions = detect_objects(frame)

                # Disegna le box sul frame
                frame_with_boxes = draw_boxes(frame, predictions)

                # Mostra il frame con le box disegnate
                cv2.imshow("Video Stream with Object Detection", frame_with_boxes)

                # Esci dal loop premendo 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Errore nella connessione o nel ricevere i frame: {e}")
                break

        cv2.destroyAllWindows()

# Esegui la connessione al WebSocket
if __name__ == "__main__":
    asyncio.run(connect_to_websocket())
