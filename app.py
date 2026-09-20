import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.inference.predict import predict_caption
from src.model.model import CNNtoRNN

app = FastAPI(title="Image Captioning API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "my_checkpoint.pth.tar")

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1

_model = None
_vocab = None


def _load_model():
    global _model, _vocab

    if _model is not None:
        return

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. Train the model first "
            f"with `python -m src.training.train`, or set the CHECKPOINT_PATH "
            f"environment variable to point at an existing checkpoint."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    vocab = checkpoint["vocab"]

    model = CNNtoRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _model, _vocab = model, vocab


@app.on_event("startup")
def startup_event():
    try:
        _load_model()
    except FileNotFoundError as e:
        print(f"[startup warning] {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    _load_model()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    caption_text = predict_caption(_model, _vocab, image, DEVICE)

    return JSONResponse({"caption": caption_text})
