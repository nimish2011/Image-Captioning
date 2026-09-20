import torchvision.transforms as transforms
import torch
from PIL import Image

def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def predict_caption(model, vocab, image: Image.Image, device):
    transform = get_transform()
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        words = model.caption_image(img_tensor, vocab)

    words = [w for w in words if w not in ("<SOS>", "<EOS>", "<PAD>")]
    return " ".join(words)