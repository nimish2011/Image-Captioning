import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


def print_examples(model, device, dataset, test_folder="test_examples"):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()

    if not os.path.isdir(test_folder):
        logger.info("'%s' not found — skipping example captions.", test_folder)
        model.train()
        return

    for fname in sorted(os.listdir(test_folder)):
        path = os.path.join(test_folder, fname)
        try:
            img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        except Exception as e:
            logger.warning("Skipping '%s': %s", fname, e)
            continue

        caption = model.caption_image(img.to(device), dataset.vocab)
        logger.info("%s -> %s", fname, " ".join(caption))

    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logger.info("Saving checkpoint to '%s'", filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    logger.info("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
