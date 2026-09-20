import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

from src.inference.predict import predict_caption
from src.logging_config import setup_logging
from src.model.model import CNNtoRNN

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_caption_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]

    model = CNNtoRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, vocab


def build_index(images_folder, checkpoint_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_caption_model(checkpoint_path, device)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    entries = []
    for fname in sorted(os.listdir(images_folder)):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue

        path = os.path.join(images_folder, fname)

        try:
            image = Image.open(path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.warning("Skipping '%s': %s", fname, e)
            continue

        caption = predict_caption(model, vocab, image, device)
        entries.append({"image": fname, "caption": caption})
        logger.info("%s -> %s", fname, caption)

    if not entries:
        raise RuntimeError(f"No readable images found in '{images_folder}'.")

    captions = [e["caption"] for e in entries]
    embeddings = embedder.encode(captions, normalize_embeddings=True)

    np.savez(
        output_path,
        images=np.array([e["image"] for e in entries]),
        captions=np.array(captions),
        embeddings=embeddings,
    )

    logger.info("Saved index for %d images to '%s'.", len(entries), output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Caption a folder of photos and build a searchable RAG index over them."
    )
    parser.add_argument("--images", default="photos", help="Folder of photos to index.")
    parser.add_argument(
        "--checkpoint", default="my_checkpoint.pth.tar", help="Trained model checkpoint."
    )
    parser.add_argument("--out", default="rag_index.npz", help="Output index file.")
    args = parser.parse_args()

    setup_logging()
    build_index(args.images, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
