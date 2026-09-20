import io
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference.predict import predict_caption
from src.logging_config import setup_logging
from src.model.model import CNNtoRNN
from src.rag.llm import generate_answer
from src.rag.retriever import CaptionRetriever

setup_logging()
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "my_checkpoint.pth.tar")
RAG_INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "rag_index.npz")

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1

_model = None
_vocab = None
_retriever = None


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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    vocab = checkpoint["vocab"]

    model = CNNtoRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _model, _vocab = model, vocab
    logger.info("Model loaded from '%s' on device '%s'.", CHECKPOINT_PATH, DEVICE)


def _load_retriever():
    global _retriever

    if _retriever is not None:
        return _retriever

    if not os.path.exists(RAG_INDEX_PATH):
        raise FileNotFoundError(
            f"RAG index not found at '{RAG_INDEX_PATH}'. Build it first with "
            f"`python -m src.rag.build_index --images <folder> --checkpoint {CHECKPOINT_PATH}`."
        )

    _retriever = CaptionRetriever(RAG_INDEX_PATH)
    logger.info("RAG index loaded from '%s'.", RAG_INDEX_PATH)
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except FileNotFoundError as e:
        logger.warning(str(e))

    try:
        _load_retriever()
    except FileNotFoundError as e:
        logger.warning(str(e))

    yield


app = FastAPI(title="Image Captioning API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "rag_index_loaded": _retriever is not None,
    }


@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        _load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        caption_text = predict_caption(_model, _vocab, image, DEVICE)
    except Exception:
        logger.exception("Caption generation failed.")
        raise HTTPException(status_code=500, detail="Caption generation failed.")

    return JSONResponse({"caption": caption_text})


@app.post("/ask")
async def ask(payload: AskRequest):
    try:
        retriever = _load_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    matches = retriever.search(payload.question, top_k=payload.top_k)
    answer = generate_answer(payload.question, matches)

    return JSONResponse({"answer": answer, "matches": matches})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
