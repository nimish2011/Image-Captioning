import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_loader.get_loader import get_loader
from src.logging_config import setup_logging
from src.model.model import CNNtoRNN
from src.training.utils import load_checkpoint, print_examples, save_checkpoint

logger = logging.getLogger(__name__)


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=0,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    writer = SummaryWriter("runs/flickr")
    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    try:
        for epoch in range(num_epochs):
            print_examples(model, device, dataset)

            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "vocab": dataset.vocab,
                }
                save_checkpoint(checkpoint)

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            for idx, (imgs, captions) in enumerate(loop):
                imgs = imgs.to(device)
                captions = captions.to(device)

                outputs = model(imgs, captions[:-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
                )

                writer.add_scalar("Training loss", loss.item(), global_step=step)
                step += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_postfix(loss=loss.item())

            logger.info("Epoch %d/%d complete.", epoch + 1, num_epochs)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user — saving current progress.")
    except Exception:
        logger.exception("Training stopped due to an unexpected error.")
        raise
    finally:
        if save_model:
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "vocab": dataset.vocab,
                }
            )
            logger.info("Final checkpoint saved.")


if __name__ == "__main__":
    setup_logging()
    train()
