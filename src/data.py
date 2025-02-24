import os
import torch
import datasets
import matplotlib.pyplot as plt
import heapq
import json


from PIL import Image
from collections import defaultdict
from umap import UMAP
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from encoders import *


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Office31EmbeddingLabelDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.num_classes = 31

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label

    

class CelebAHQDataset(Dataset):
    def __init__(self, split="train", transform=None, dino_transform=None):
        self.dataset = datasets.load_dataset("cld07/captioned_ffhq_20k_512")[split]
        # self.dataset = datasets.load_dataset("cld07/captioned_ffhq_512")[split]
        self.transform = transform
        self.dino_transform = dino_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['source_image'].convert('RGB')
        caption = item['caption']
        
        # Return both transformed versions if dino_transform is provided
        if self.dino_transform:
            gan_image = self.transform(image) if self.transform else image
            dino_image = self.dino_transform(image)
            return gan_image, dino_image, caption
        else:
            # Return single transformed image for reconstruction
            if self.transform:
                image = self.transform(image)
            return image, caption
        

class ZeroShotImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the class folders
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create text labels for zero-shot classification
        self.text_labels = [f"a photo of {cls.lower()}" for cls in self.classes]
        
        # Collect all image paths
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Transformed image
            label (int): Numeric class label
            text_label (str): Text description for zero-shot classification
        """
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        text_label = self.text_labels[label]
        
        return image, label, text_label



# Example usage:
if __name__ == "__main__":
    # Initialize dataset
    dataset = ZeroShotImageDataset(root_dir="path/to/your/image/folder")
    
    # Access the class names and their text labels
    print("Classes:", dataset.classes)
    print("Text labels:", dataset.text_labels)
    
    # Get a sample
    image, label, text_label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Text description: {text_label}")


class NoisyMNIST(Dataset):
    def __init__(self, path):
        data = np.load(path)
        view1 = data["view_0"].reshape(70000, -1)
        view2 = data["view_1"].reshape(70000, -1)
        self.n_samples = len(view1)

        self.views = [torch.from_numpy(view1), torch.from_numpy(view2)]
        self.labels = torch.from_numpy(data["labels"])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return [view[idx] for view in self.views], self.labels[idx]
    

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all files from domain A
        self.files_A = sorted(os.listdir(os.path.join(root_dir, 'train_A')))
        self.files_B = sorted(os.listdir(os.path.join(root_dir, 'train_B')))
        
        assert len(self.files_A) == len(self.files_B), "Mismatch in number of files between domains"
        
    def __len__(self):
        return len(self.files_A)
    
    def __getitem__(self, idx):
        img_A_path = os.path.join(self.root_dir, 'train_A', self.files_A[idx])
        img_B_path = os.path.join(self.root_dir, 'train_B', self.files_B[idx])
        
        image_A = Image.open(img_A_path).convert('RGB')
        image_B = Image.open(img_B_path).convert('RGB')
        
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
        
        return {
            'A': image_A,
            'B': image_B,
            'name': os.path.splitext(self.files_A[idx])[0]
        }
    

class FashionDataset(Dataset):
    def __init__(self, num_samples=20000):
        self.dataset = datasets.load_dataset("Marqo/polyvore")['data']
        image_encoder = FashionImageEncoder()
        self.transform = image_encoder.preprocess
        
    def __len__(self):
        return 20000
    
    def __getitem__(self, idx):
        # Get image - the image is already a PIL Image
        image = self.dataset[idx]['image']
        image = self.transform(image)  # Just apply the transform
        
        # Get text (category 3)
        text = self.dataset[idx]['text']
        
        return image, text

def ensure_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


class MSCOCODataset(Dataset):
    def __init__(self, images_dir='../data/coco/images', captions_file='../data/coco/captions.json'):
        # Load caption data
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.caption_data = json.load(f)
        
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Lambda(ensure_rgb),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.caption_data)
    
    def __getitem__(self, idx):
        item = self.caption_data[idx]
        image_path = f"../data/coco/{item['image_filename']}"
        caption = item['caption']
        
        try:
            # Load and transform image
            with Image.open(image_path) as img:
                image_tensor = self.transform(img)
            return image_tensor, caption
        except Exception as e:
            print(f"Error loading image at {image_path}: {str(e)}")
            return torch.zeros(3, 224, 224), caption
    
    def get_original_item(self, idx):
        item = self.caption_data[idx]
        return {
            'image': Image.open(item['image_filename']),
            'label': item['caption']
        }
        


class Flickr(Dataset):
    def __init__(self, root: str, ann_file: str, transform=None):
        super().__init__()

        self.root = root
        self.ann_file = ann_file
        self.transform = transform

        self.annotations = {}
        with open(ann_file, "r") as f:
            f.readline()
            for line in f:
                decomposed = line.strip().split(",")
                img_id, caption = decomposed[0], decomposed[1]
                if img_id not in self.annotations:
                    self.annotations[img_id] = caption

        # State variables
        self.in_table = False
        self.current_tag = None
        self.current_img = None

    def __len__(self):
        return len(self.annotations.keys())

    def __getitem__(self, idx):
        img_id = list(self.annotations.keys())[idx]
        caption = self.annotations[img_id]
        img = Image.open(self.root + "/" + img_id).convert("RGB")
        if self.transform:
            img = self.transform(img).float()

        return img, caption
    

def plot_neighbor_images(image_idx, nn_idxs, dataset, path, max_neighbors=10):
    num_neighbors = min(len(nn_idxs), max_neighbors)
    fig, axes = plt.subplots(num_neighbors + 1, 1, figsize=(5, 5 * (num_neighbors + 1)))

    image = dataset[image_idx][0].permute(1, 2, 0)
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, idx in enumerate(nn_idxs):
        # if i >= 100 and i < 110:
        if i == 10:
            break
        neighbor = dataset[idx][0].permute(1, 2, 0)
        axes[i + 1].imshow(neighbor)
        axes[i + 1].set_title(f"Neighbor {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{path}/image_neighbors_{image_idx}.png", bbox_inches="tight")
    plt.close()


def plot_neighbor_images_and_captions(
    image, caption, image_neighbors, captions_neighbors
):
    num_neighbors = len(image_neighbors)
    fig, axes = plt.subplots(num_neighbors + 1, 1, figsize=(5, 5 * (num_neighbors + 1)))

    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, (img, cpt) in enumerate(image_neighbors):
        axes[i + 1].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i + 1].set_title(f"Neighbor {i+1}")
        axes[i + 1].axis("off")
        axes[i + 1].annotate(
            cpt,
            xy=(0.5, -0.05),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("image_neighbors.png", bbox_inches="tight")
    plt.show()
    plt.close()

    print("Caption: ", caption)
    print("Neighbor Captions:")
    for i, caption in enumerate(captions_neighbors):
        print(f"Neighbor {i+1}: {caption}")


def find_k_nearest_neighbors(embeddings, k):
    # distances = torch.cdist(embeddings, embeddings)
    distances = 1 - torch.mm(embeddings, embeddings.t())
    neighbor_distances, neighbor_indices = torch.topk(distances, k=k + 1, largest=False)
    neighbor_distances = neighbor_distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]
    return neighbor_indices, neighbor_distances


def plot_image_text_similarities(dataset, encoded1, encoded2):
    image_nn_indices, image_nn_dis = find_k_nearest_neighbors(encoded1, 200)
    text_nn_indices, text_nn_dis = find_k_nearest_neighbors(encoded2, 200)
    flickr = dataset
    image = flickr[10][0]
    image_nearest_neighbors = [
        (flickr[idx][0], flickr[idx][1]) for idx in image_nn_indices[10, :10]
    ]
    text = flickr[10][1]
    text_nearest_neighbors = [flickr[idx][1] for idx in text_nn_indices[10, :10]]
    plot_neighbor_images_and_captions(
        image, text, image_nearest_neighbors, text_nearest_neighbors
    )


def create_weakly_parallel_data(dataset, n_parallel, removal_percentage=0.1, random_state=42):
    encoded1, encoded2 = dataset
    n_total = encoded1.size(0)
    n_unparallel = n_total - n_parallel


    if n_parallel > n_total:
        raise ValueError("n_parallel cannot be larger than the dataset size")

    n_remove = int(removal_percentage * n_unparallel)

    # Generate random indices to remove from encoded1 and encoded2 independently
    indices_to_remove_1 = torch.randperm(n_unparallel)[:n_remove]
    indices_to_remove_2 = torch.randperm(n_unparallel)[:n_remove]

    # Remove selected indices from encoded1 and encoded2
    mask1 = torch.ones(n_unparallel, dtype=torch.bool)
    mask1[indices_to_remove_1] = False
    encoded1_unparallel = encoded1[:n_unparallel][mask1]

    mask2 = torch.ones(n_unparallel, dtype=torch.bool)
    mask2[indices_to_remove_2] = False
    encoded2_unparallel = encoded2[:n_unparallel][mask2]

    # shuffle the second mdality to remove any aligment in the data
    encoded2_unparallel = encoded2_unparallel[
        torch.randperm(encoded2_unparallel.size(0))
    ]

    encoded1_parallel = encoded1[n_unparallel:]  # Last n_parallel part of encoded1
    encoded2_parallel = encoded2[n_unparallel:]  # Last n_parallel part of encoded2

    # Concatenate unparallel and parallel portions back together
    encoded1_weak = torch.cat((encoded1_unparallel, encoded1_parallel), dim=0)
    encoded2_weak = torch.cat((encoded2_unparallel, encoded2_parallel), dim=0)

    return (encoded1_weak, encoded2_weak)


def create_unparallel_parallel_data(dataset):
    create_weakly_parallel_data(dataset, 0)


def train_test_split(encoded1, encoded2, n_test=400, dataset_name=None, random_state=42):
    if encoded1.size(0) != encoded2.size(0):
        raise ValueError(
            "The two encoded modalities must have the same number of samples"
        )

    
    n_samples = encoded1.size(0)
    indices = range(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    test_set = (encoded1[test_indices], encoded2[test_indices])
    train_set = (encoded1[train_indices], encoded2[train_indices])

    return train_set, test_set


def embed_raw_data_using_pretrained_encoders(dataset, encoder1, encoder2):
    batch_size = 128
    parallel_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder1, encoder2 = encoder1.to(device), encoder2.to(device)

    for i, (first, second) in enumerate(parallel_loader):
        with torch.no_grad():
            if i == 0:
                first_embeddings = encoder1(first, device=device)
                second_embeddings = encoder2(second, device=device)
            else:
                first_embeddings = torch.cat(
                    (first_embeddings, encoder1(first, device=device)), dim=0
                )
                second_embeddings = torch.cat(
                    (second_embeddings, encoder2(second, device=device)), dim=0
                )

    return first_embeddings, second_embeddings



def preprocess_data(path_to_encodings, n_test=400):
    try:
        encoded1 = torch.load(os.path.join(path_to_encodings, "encoded1.pt"))
        encoded2 = torch.load(os.path.join(path_to_encodings, "encoded2.pt"))
    except FileNotFoundError:
        raise FileNotFoundError("Can't find the encoded data. Please first encode the data.")
    
    print("Splitting the data into train and test...")
    train_set, test_set = train_test_split(encoded1, encoded2, n_test=n_test)
    return train_set, test_set



def load_dataset(dataset_name, n_test):
    if dataset_name == "flickr30":
        train_set, test_set = preprocess_data(path_to_encodings="../data/flickr30/", n_test=n_test)
        return train_set, test_set

    else:
        raise ValueError("Dataset is not supported, please add this dataset manually.")
