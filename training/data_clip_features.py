import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class ClipFeatureDataset(Dataset):
    """Dataset returning CLIP patch embeddings and first-layer features."""

    def __init__(self, split="train", data_root="./data", num_samples=None):
        self.dataset = CIFAR10(root=data_root, train=split == "train", download=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).vision_model
        self.vision_model.eval()
        self.num_samples = num_samples

    def __len__(self):
        if self.num_samples is not None:
            return min(self.num_samples, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.fromarray(image)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_model(
                pixel_values=inputs["pixel_values"], output_hidden_states=True
            )
        patch = outputs.hidden_states[0][0]
        feature = outputs.hidden_states[1][0]
        return patch, feature, label

