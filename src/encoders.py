import torch
import numpy as np
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2FeatureExtractor, HubertModel


import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import open_clip



class FashionImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        
    def forward(self, input, device):
        self.model = self.model.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(input)
            
        return image_features

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = torch.hub.load("facebookresearch/vicreg:main", "resnet50")
        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    def forward(self, input, device):
        input = input.to(device)
        return self.encoder(input)
    
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def forward(self, input, device):
        encoded = self.model.encode(input, device=device)
        encoded = np.stack(encoded)
        encoded = torch.from_numpy(encoded).float().to(device)
        encoded = nn.functional.pad(encoded, (0, 2048 - 384))
        return encoded

