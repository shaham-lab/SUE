import torch
import open_clip
import numpy as np
import matplotlib.pyplot as plt

from data import *
from encoders import *
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_distances
from sklearn.random_projection import GaussianRandomProjection
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F


import pickle

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def plot_retrieved_images(indices, dataset, query_caption, save_path=None):
    indices = [int(i) for i in indices]
    n = len(indices)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    plt.suptitle(f'Retrieved images for caption: "{query_caption}"', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            image = dataset[indices[int(idx)]][0]  # Assuming dataset has get_image method
            # image = tensor2im(image)
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Image {indices[int(idx)]}',fontsize=10, wrap=True)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)


def compute_whitening_matrix(cov):
    S, V = np.linalg.eig(cov)
    S_sqrt_inv = np.diag(1 / np.sqrt(S + 1e-6))
    W = V @ S_sqrt_inv @ V.T
    return W


def compute_local_covariance(sample, data, top_k=50, n_iter=5):
    cov = np.eye(data.shape[1])

    for _ in range(n_iter):
        W = compute_whitening_matrix(cov)

        data_transformed = data @ W
        sample_transformed = sample.reshape(1, -1) @ W

        nbrs = NearestNeighbors(
            n_neighbors=top_k, algorithm="auto", metric="cosine"
        ).fit(data_transformed)

        _, indices = nbrs.kneighbors(sample_transformed)

        x_original = data[indices[0]]
        cov = np.cov(x_original.T)

    return cov


def compute_affinity_matrix_with_local_mahalanobis(X, top_k=200, n_iter=8, target_dim=70):
    """Compute affinity matrix using local Mahalanobis distances for refined nearest neighbors."""
    
    transformer = GaussianRandomProjection(n_components=target_dim, random_state=42)
    X_reduced = transformer.fit_transform(X)

    N = X.shape[0]
    affinity_matrix = np.zeros((N, N))

    # Find initial neighbors using cosine distance
    nn_initial = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(X_reduced)
    _, ngb_indices = nn_initial.kneighbors(X_reduced)

    for i, sample in enumerate(X):
        print(f"Processing sample {i}/{N}")
        initial_data_reduced = X_reduced[ngb_indices[i]]

        # Compute local covariance
        local_cov = compute_local_covariance(
            X_reduced[i],
            data=initial_data_reduced,
            top_k=top_k,
            n_iter=n_iter,
        )

        # Compute whitening matrix
        W = compute_whitening_matrix(cov=local_cov)
        whitened_data = initial_data_reduced @ W

        # Find refined neighbors in whitened space
        nn_final = NearestNeighbors(n_neighbors=int(top_k * 0.3), metric="cosine").fit(
            whitened_data
        )
        _, ngb_indices_final = nn_final.kneighbors(whitened_data[0].reshape(1, -1))
        ngb_indices_final = ngb_indices_final[:, 1:]  # Remove self from neighbors

        # Get indices of final neighbors in original space
        final_neighbors_indices = ngb_indices[i][ngb_indices_final[0]]

        # Compute Mahalanobis distances for final neighbors
        whitened_center = X_reduced[i] @ W
        whitened_final_neighbors = X_reduced[final_neighbors_indices] @ W
        
        mahalanobis_distances = cosine_distances(
            whitened_center.reshape(1, -1),
            whitened_final_neighbors
        )[0]

        # Store the Mahalanobis distances for final neighbors
        affinity_matrix[i, final_neighbors_indices] = mahalanobis_distances

    return (affinity_matrix + affinity_matrix.T) / 2


def compute_affinity_matrix_with_local_mahalanobis_rbf(
    X, top_k=200, n_iter=8, sigma=1.0, target_dim=70
):

    transformer = GaussianRandomProjection(n_components=target_dim, random_state=42)
    X_reduced = transformer.fit_transform(X)

    N = X.shape[0]
    affinity_matrix = np.zeros((N, N))

    nn_initial = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(X_reduced)
    _, ngb_indices = nn_initial.kneighbors(X_reduced)

    for i, sample in enumerate(X):
        print(f"Processing sample {i}/{N}")
        initial_data_reduced = X_reduced[ngb_indices[i]]

        local_cov = compute_local_covariance(
            X_reduced[i],
            data=initial_data_reduced,
            top_k=top_k,
            n_iter=n_iter,
        )

        W = compute_whitening_matrix(cov=local_cov)
        whitened_data = initial_data_reduced @ W

        nn_final = NearestNeighbors(n_neighbors=int(top_k * 0.3), metric="cosine").fit(
            whitened_data
        )
        _, ngb_indices_final = nn_final.kneighbors(whitened_data[0].reshape(1, -1))
        ngb_indices_final = ngb_indices_final[:, 1:]

        final_neighbors_indices = ngb_indices[i][ngb_indices_final[0]]

        original_distances = cosine_distances(
            X[i].reshape(1, -1), X[final_neighbors_indices]
        )[0]
        original_distances = original_distances - np.min(original_distances)
        sigma = np.median(original_distances)

        # Compute RBF affinities using original distances
        rbf_affinities = np.exp(-(original_distances**2) / (2 * sigma**2))
        affinity_matrix[i, final_neighbors_indices] = rbf_affinities

    return (affinity_matrix + affinity_matrix.T) / 2


def custom_cosine_rbf_affinity(X, top_k=100):
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances - distances[:, [0]]

    scale = np.median([
       np.median(dist[dist > 0]) for dist in distances
   ])

    n_samples = X.shape[0]
    affinity_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j, idx in enumerate(indices[i]):
            cosine_similarity = 1 - distances[i][j]
            affinity_matrix[i, idx] = np.exp(-((1 - cosine_similarity) ** 2) / (2 * scale ** 2))

    return (affinity_matrix + affinity_matrix.T) / 2


def cosine_affinity(X, top_k=100):
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Convert cosine distances to similarities
    similarities = 1 - distances
    
    # Initialize affinity matrix
    n_samples = X.shape[0]
    affinity_matrix = np.zeros((n_samples, n_samples))
    
    # Fill in similarities for nearest neighbors
    for i in range(n_samples):
        for j, idx in enumerate(indices[i]):
            affinity_matrix[i, idx] = similarities[i][j]
    
    # Make matrix symmetric by averaging with its transpose
    return (affinity_matrix + affinity_matrix.T) / 2


def dot_product_affinity(X, top_k=100):
    # Compute all pairwise dot products
    dot_products = np.dot(X, X.T)
    
    n_samples = X.shape[0]
    # For each point, find indices of top_k largest dot products
    indices = np.argpartition(-dot_products, kth=top_k, axis=1)[:, :top_k]
    
    affinity_matrix = np.zeros((n_samples, n_samples))
    
    # Fill affinity matrix with dot products only for nearest neighbors
    for i in range(n_samples):
        neighbors = indices[i]
        affinity_matrix[i, neighbors] = dot_products[i, neighbors]
    
    # Make the matrix symmetric by averaging with its transpose
    return (affinity_matrix + affinity_matrix.T) / 2


def compute_normalized_laplacian(affinity_matrix):
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix))
    identity_matrix = np.eye(affinity_matrix.shape[0])
    laplacian_norm = (
        identity_matrix - degree_inv_sqrt @ affinity_matrix @ degree_inv_sqrt
    )
    return laplacian_norm


def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    best_alignment_indices = None
    best_alignment_score = 0

    for i in range(-1, x_feats.shape[1]):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
                    
            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            
            score = mutual_knn(x, y, topk)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    
    return best_alignment_score, best_alignment_indices


def mutual_knn(feats_A, feats_B, topk):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    # L2 normalization
    feats_A = F.normalize(feats_A, p=2, dim=-1)
    feats_B = F.normalize(feats_B, p=2, dim=-1)

    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)   

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0
    
    acc = (lvm_mask * llm_mask).sum(dim=1) / topk
    
    return acc.mean().item()
    

def measure_overlap(train_modality1, train_modality2, top_k=100):
    nn_modality1 = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(
        train_modality1
    )
    _, indices_modality1 = nn_modality1.kneighbors(train_modality1)

    nn_modality2 = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(
        train_modality2
    )
    _, indices_modality2 = nn_modality2.kneighbors(train_modality2)

    overlap_ratios = []
    n_samples = train_modality1.shape[0]

    for i in range(n_samples):
        neighbors_modality1 = set(indices_modality1[i])
        neighbors_modality2 = set(indices_modality2[i])

        intersection = len(neighbors_modality1.intersection(neighbors_modality2))
        union = len(neighbors_modality1.union(neighbors_modality2))

        if union == 0:
            jaccard_similarity = 0
        else:
            jaccard_similarity = intersection / union

        overlap_ratios.append(jaccard_similarity)
    return overlap_ratios


def get_spectral_embedding(X: np.ndarray, n_components: int = 2, n_neighbors: int = 10) -> np.ndarray:
    W = custom_cosine_rbf_affinity(X, n_neighbors)
    
    D_inv = np.diag(1 / np.sum(W, axis=1))
    
    L_rw = np.eye(D_inv.shape[0]) - D_inv @ W
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_rw)
    
    idx = np.argsort(eigenvalues)  # ascending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    embedding = eigenvectors[:, 1:n_components + 1]
    return embedding


def align_modalities_unpaired(X1: np.ndarray, X2: np.ndarray, batch_size: int = None, repeats: int = 1) -> np.ndarray:
    """Aligns the SE embeddings using LC"""
    if batch_size is None:
        batch_size = X1.shape[0]

    coefs = np.ones((repeats, X1.shape[1]))
    for i in range(repeats):
        coefs[i] = align_modalities_unpaired_once(X1, X2, batch_size)

    coefs = np.sign(coefs.sum(axis=0))

    return coefs


def align_modalities_unpaired_once(X1: np.ndarray, X2: np.ndarray, batch_size: int) -> np.ndarray:
    """Aligns the SE embeddings using LC"""
    # choose batch_size random samples
    inds1 = np.random.choice(X1.shape[0], batch_size, replace=False)
    inds2 = np.random.choice(X2.shape[0], batch_size, replace=False)
    X1_batch, X2_batch = X1[inds1], X2[inds2]

    v = np.ones((X1_batch.shape[0],))
    v /= np.linalg.norm(v)

    # normalize the embeddings
    X1_batch = X1_batch / np.linalg.norm(X1_batch, axis=0)
    X2_batch = X2_batch / np.linalg.norm(X2_batch, axis=0)

    coefs = np.ones((X1.shape[1],))
    for i in range(X1.shape[1]):
        angle_X1 = v.dot(X1_batch[:, i])
        angle_X2 = v.dot(X2_batch[:, i])

        # print(angle_X1, angle_X2)

        if np.sign(angle_X1) != np.sign(angle_X2):
            coefs[i] = -1

    return coefs


def compute_clip_scores(embedding1, embedding2, mode='i2t', device: str = "cuda:1", dataset_name: str = 'flickr30'):
    # Input validation
    if mode not in ['i2t', 't2i']:
        raise ValueError("Mode must be either 'i2t' or 't2i'")
    if dataset_name not in ['flickr8', 'flickr30', 'fashion', 'coco']:
        raise ValueError("Dataset must be either 'flickr8' or 'flickr30'")

    # Model setup
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Dataset setup

    if dataset_name == 'fashion':
        dataset = FashionDataset()
    elif dataset_name == 'coco':
        dataset = MSCOCODataset()
    else:
        dataset = Flickr(
                root=f"../data/{dataset_name}/images",
                ann_file=f"../data/{dataset_name}/captions.txt",
                transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )



    # Prepare embeddings
    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity matrix based on mode
    cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)

    # Get top 1 retrieval
    _, indices = torch.topk(cosine_similarity_matrix, largest=True, k=1)
    
    relative_clip_scores = []
    query_indices = range(cosine_similarity_matrix.shape[0])
    
    def process_image(image_tensor):
        """Helper function to process image tensor to PIL image"""
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                   np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def compute_clip_score(image_pil, caption, model, processor, device):
        """Helper function to compute CLIP score for an image-caption pair"""
        inputs = processor(
            images=image_pil,
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine = torch.matmul(image_features, text_features.T)
            clip_score = torch.max(torch.tensor(100.0) * cosine, torch.tensor(0.0))
            
        return clip_score

    for query_idx in query_indices:
        if mode == 'i2t':
            # Image to text mode
            image = dataset[query_idx][0]
            original_caption = dataset[query_idx][1]
            retrieved_caption = dataset[int(indices[query_idx].item())][1]
            
            image_pil = process_image(image)
            
            # Compute scores
            original_score = compute_clip_score(image_pil, original_caption, model, processor, device)
            retrieved_score = compute_clip_score(image_pil, retrieved_caption, model, processor, device)
            
        else:
            # Text to image mode
            query_caption = dataset[query_idx][1]
            original_image = dataset[query_idx][0]
            retrieved_image = dataset[int(indices[query_idx].item())][0]
            
            original_image_pil = process_image(original_image)
            retrieved_image_pil = process_image(retrieved_image)
               
            # Compute scores
            original_score = compute_clip_score(original_image_pil, query_caption, model, processor, device)
            retrieved_score = compute_clip_score(retrieved_image_pil, query_caption, model, processor, device)
            

        # Compute relative score
        relative_score = retrieved_score / original_score
        relative_clip_scores.append(relative_score.item())

    relative_scores_array = np.array(relative_clip_scores)
    return (np.mean(relative_scores_array), np.std(relative_scores_array))


def compute_clip_scores_k(embedding1, embedding2, mode='i2t', device: str = "cuda:1", dataset_name: str = 'flickr30'):
    # Input validation
    if mode not in ['i2t', 't2i']:
        raise ValueError("Mode must be either 'i2t' or 't2i'")
    if dataset_name not in ['flickr8', 'flickr30']:
        raise ValueError("Dataset must be either 'flickr8' or 'flickr30'")

    # Model setup
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Dataset setup
    dataset = Flickr(
        root=f"../data/{dataset_name}/images",
        ann_file=f"../data/{dataset_name}/captions.txt",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    # Prepare embeddings
    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity matrix
    cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)

    # Get top-k retrieval for k=1, 5, 10
    k_values = [1, 5, 10]
    top_k_indices = {k: torch.topk(cosine_similarity_matrix, k=k, dim=1).indices for k in k_values}
    
    relative_clip_scores = {k: [] for k in k_values}
    
    def process_image(image_tensor):
        """Helper function to process image tensor to PIL image"""
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                   np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def compute_clip_score(image_pil, caption, model, processor, device):
        """Helper function to compute CLIP score for an image-caption pair"""
        inputs = processor(
            images=image_pil,
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine = torch.matmul(image_features, text_features.T)
            
        return torch.clamp(cosine, min=0.0, max=1.0).item()

    for query_idx in range(cosine_similarity_matrix.shape[0]):
        if mode == 'i2t':
            image = dataset[query_idx][0]
            original_caption = dataset[query_idx][1]
            image_pil = process_image(image)
        else:
            query_caption = dataset[query_idx][1]

        for k in k_values:
            retrieved_scores = []
            for neighbor_idx in top_k_indices[k][query_idx]:
                if mode == 'i2t':
                    retrieved_caption = dataset[int(neighbor_idx)][1]
                    score = compute_clip_score(image_pil, retrieved_caption, model, processor, device)
                else:
                    retrieved_image = dataset[int(neighbor_idx)][0]
                    retrieved_image_pil = process_image(retrieved_image)
                    score = compute_clip_score(retrieved_image_pil, query_caption, model, processor, device)
                
                retrieved_scores.append(score)

            if mode == 'i2t':
                original_score = compute_clip_score(image_pil, original_caption, model, processor, device)
            else:
                original_image = dataset[query_idx][0]
                original_image_pil = process_image(original_image)
                original_score = compute_clip_score(original_image_pil, query_caption, model, processor, device)

            # Compute relative score for this k
            average_retrieved_score = sum(retrieved_scores) / k
            relative_score = average_retrieved_score / original_score
            relative_clip_scores[k].append(relative_score)

    # Calculate final averages and standard deviations for each k
    results = {}
    for k in k_values:
        scores_array = np.array(relative_clip_scores[k])
        results[f"top-{k}"] = {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array)
        }

    # Print and return results
    for k in k_values:
        print(f"Top-{k} relative CLIP score: mean={results[f'top-{k}']['mean']}, std={results[f'top-{k}']['std']}")

    return results


def compute_clip_scores_5_10(embedding1, embedding2, mode='i2t', device: str = "cuda:1", dataset_name: str = 'flickr30'):
    if mode not in ['i2t', 't2i']:
        raise ValueError("Mode must be either 'i2t' or 't2i'")
    if dataset_name not in ['flickr8', 'flickr30']:
        raise ValueError("Dataset must be either 'flickr8' or 'flickr30'")

    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    dataset = Flickr(
        root=f"../data/{dataset_name}/images",
        ann_file=f"../data/{dataset_name}/captions.txt",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)
    
    # Get top-10 indices and select only positions 5-10
    top_10_indices = torch.topk(cosine_similarity_matrix, k=10, dim=1).indices
    indices_5_10 = top_10_indices[:, 4:10]  # Zero-based indexing
    
    relative_clip_scores = []
    
    def process_image(image_tensor):
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                   np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def compute_clip_score(image_pil, caption, model, processor, device):
        inputs = processor(
            images=image_pil,
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine = torch.matmul(image_features, text_features.T)
            
        return torch.clamp(cosine, min=0.0, max=1.0).item()

    for query_idx in range(cosine_similarity_matrix.shape[0]):
        if mode == 'i2t':
            image = dataset[query_idx][0]
            original_caption = dataset[query_idx][1]
            image_pil = process_image(image)
        else:
            query_caption = dataset[query_idx][1]

        retrieved_scores = []
        for neighbor_idx in indices_5_10[query_idx]:
            if mode == 'i2t':
                retrieved_caption = dataset[int(neighbor_idx)][1]
                score = compute_clip_score(image_pil, retrieved_caption, model, processor, device)
            else:
                retrieved_image = dataset[int(neighbor_idx)][0]
                retrieved_image_pil = process_image(retrieved_image)
                score = compute_clip_score(retrieved_image_pil, query_caption, model, processor, device)
            
            retrieved_scores.append(score)

        if mode == 'i2t':
            original_score = compute_clip_score(image_pil, original_caption, model, processor, device)
        else:
            original_image = dataset[query_idx][0]
            original_image_pil = process_image(original_image)
            original_score = compute_clip_score(original_image_pil, query_caption, model, processor, device)

        average_retrieved_score = sum(retrieved_scores) / len(retrieved_scores)
        relative_score = average_retrieved_score / original_score
        relative_clip_scores.append(relative_score)

    scores_array = np.array(relative_clip_scores)
    results = {
        "neighbors_5_10": {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array)
        }
    }

    print(f"Neighbors 5-10 relative CLIP score: mean={results['neighbors_5_10']['mean']}, std={results['neighbors_5_10']['std']}")

    return results


def visual_evaluate(embedding1, embedding2, mode='i2t', dataset_name='flickr8', num_samples=40, k=10, output_dir="visual_results"):
    if mode not in ['i2t', 't2i']:
        raise ValueError("Mode must be either 'i2t' or 't2i'")
    
    if dataset_name not in ['flickr8', 'flickr30', 'fashion', 'coco']:
        raise ValueError("Dataset must be either 'flickr8' or 'flickr30'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset_name == 'fashion':
        dataset = FashionDataset()
    elif dataset_name == 'coco':
        dataset = MSCOCODataset(
            images_dir='../data/coco/images',
            captions_file='../data/coco/captions.json'
        )
    else:
        dataset = Flickr(
                root=f"../data/{dataset_name}/images",
                ann_file=f"../data/{dataset_name}/captions.txt",
                transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )

    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    if mode == 'i2t':
        cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)
    else:  # t2i
        cosine_similarity_matrix = torch.matmul(embedding2, embedding1.T)

    _, indices = torch.topk(cosine_similarity_matrix, largest=True, k=k)
    
    def process_image(image_tensor):
        """Helper function to process image tensor to numpy array"""
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                       np.array([0.485, 0.456, 0.406]))
            image_np = np.clip(image_np, 0, 1)
            return image_np
        return image_tensor

    for idx in range(num_samples):
        if mode == 'i2t':
            # Image-to-text visualization
            image = dataset[int(idx)][0]
            image_np = process_image(image)
            
            # Plot and save query image
            plt.figure(figsize=(10, 10))
            plt.imshow(image_np)
            plt.axis('off')
            plt.savefig(f"{output_dir}/image_{idx}.png", bbox_inches='tight')
            plt.close()

            # Write captions to file
            with open(f"{output_dir}/captions_{idx}.txt", "w", encoding='utf-8') as caption_file:
                caption_file.write(f"Original Caption: {dataset[int(idx)][1]}\n\n")
                
                for n in range(k):
                    retrieved_idx = int(indices[idx, n].item())
                    caption = dataset[retrieved_idx][1]
                    caption_file.write(f"Top {n+1} caption: {caption}\n")
                    
        else:  # t2i mode
            # Text-to-image visualization
            query_caption = dataset[idx][1]
            
            # Create grid of retrieved images
            rows = cols = int(np.ceil(np.sqrt(k)))
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            fig.suptitle(f'Query Caption: {query_caption}', wrap=True, fontsize=16)
            
            # In case k=1, make axes indexable
            if k == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes[np.newaxis, :]
            elif cols == 1:
                axes = axes[:, np.newaxis]
            
            for i, ax in enumerate(axes.flat):
                if i < k:  # Only process up to k images
                    img_idx = int(indices[idx, i].item())
                    image = dataset[img_idx][0]
                    image_np = process_image(image)
                    
                    ax.imshow(image_np)
                    ax.axis('off')
                else:  # Hide empty subplots
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/caption_{idx}_nearest_images.png", 
                       bbox_inches='tight', pad_inches=0.2)
            plt.close()

    print(f"Visual evaluation results saved to {output_dir}")


def visual_evaluate_pix2pix(embedding1, embedding2, mode='edge2real', dataset_path='../data/pix2pix/edges2shoes', num_samples=50, k=10, output_dir="visual_results"):
    if mode not in ['edge2real', 'real2edge']:
        raise ValueError("Mode must be either 'edge2real' or 'real2edge'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = Pix2PixDataset(dataset_path, transform=transform)

    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    # Normalize embeddings
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    if mode == 'edge2real':
        cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)
    else:  # real2edge
        cosine_similarity_matrix = torch.matmul(embedding2, embedding1.T)

    _, indices = torch.topk(cosine_similarity_matrix, largest=True, k=k)
    
    def process_image(image_tensor):
        """Helper function to process image tensor to numpy array"""
        if isinstance(image_tensor, torch.Tensor):
            # Denormalize
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                       np.array([0.485, 0.456, 0.406]))
            image_np = np.clip(image_np, 0, 1)
            return image_np
        return image_tensor

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        
        if mode == 'edge2real':
            # Edge-to-real visualization
            query_image = sample['A']  # Edge image
            query_image_np = process_image(query_image)
            
            # Create visualization grid
            rows = cols = int(np.ceil(np.sqrt(k + 1)))  # +1 for query image
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            
            # Make axes indexable for all cases
            if not isinstance(axes, np.ndarray):
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes[np.newaxis, :]
            
            # Plot query image
            axes[0, 0].imshow(query_image_np)
            axes[0, 0].set_title('Query (Edge)', fontsize=10)
            axes[0, 0].axis('off')
            
            # Plot retrieved images
            for i, ax in enumerate(axes.flat[1:], 1):
                if i <= k:
                    retrieved_idx = int(indices[idx, i-1].item())
                    retrieved_image = dataset[retrieved_idx]['B']  # Real image
                    retrieved_image_np = process_image(retrieved_image)
                    
                    ax.imshow(retrieved_image_np)
                    ax.set_title(f'Top {i}', fontsize=10)
                    ax.axis('off')
                else:
                    ax.axis('off')
                    
        else:  # real2edge mode
            # Real-to-edge visualization
            query_image = sample['B']  # Real image
            query_image_np = process_image(query_image)
            
            # Create visualization grid
            rows = cols = int(np.ceil(np.sqrt(k + 1)))  # +1 for query image
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            
            # Make axes indexable for all cases
            if not isinstance(axes, np.ndarray):
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes[np.newaxis, :]
            
            # Plot query image
            axes[0, 0].imshow(query_image_np)
            axes[0, 0].set_title('Query (Real)', fontsize=10)
            axes[0, 0].axis('off')
            
            # Plot retrieved images
            for i, ax in enumerate(axes.flat[1:], 1):
                if i <= k:
                    retrieved_idx = int(indices[idx, i-1].item())
                    retrieved_image = dataset[retrieved_idx]['A']  # Edge image
                    retrieved_image_np = process_image(retrieved_image)
                    
                    ax.imshow(retrieved_image_np)
                    ax.set_title(f'Top {i}', fontsize=10)
                    ax.axis('off')
                else:
                    ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/query_{idx}_nearest_images_{mode}.png", 
                   bbox_inches='tight', pad_inches=0.2)
        plt.close()

    print(f"Visual evaluation results saved to {output_dir}")


def compute_soft_retrieval_t2i(final_embedding1, final_embedding2, device="cuda", dataset='flickr8'):
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    dino.eval()
    
    print(f"Computing soft retrieval scores for {dataset}")
    
    if dataset == 'flickr8' or dataset == 'flickr30':
        dataset = Flickr(
            root=f"../data/{dataset}/images",
            ann_file=f"../data/{dataset}/captions.txt",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    else:
        raise ValueError("Not supported dataset")
    
    # Convert to torch tensors and move to device
    final_embedding1 = torch.from_numpy(final_embedding1).float().to(device)
    final_embedding2 = torch.from_numpy(final_embedding2).float().to(device)

    # Normalize embeddings
    final_embedding1 = final_embedding1 / final_embedding1.norm(p=2, dim=-1, keepdim=True)
    final_embedding2 = final_embedding2 / final_embedding2.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity matrix and get top-1 indices (t2i, so text @ image.T)
    cosine_similarity_matrix = final_embedding2 @ final_embedding1.T
    _, indices = torch.topk(cosine_similarity_matrix, largest=True, k=1)

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])

    all_soft_scores = []
    
    with torch.no_grad():
        for caption_idx in range(len(dataset)):
            # Get original image for this caption
            original_image = dataset[caption_idx][0].to(device)
            
            # Preprocess image
            original_tensor = preprocess(original_image).unsqueeze(0)
            original_dino_emb = dino(original_tensor)
            
            # Get retrieved image
            retrieved_idx = indices[caption_idx].item()
            retrieved_image = dataset[retrieved_idx][0].to(device)
            
            # Preprocess retrieved image
            retrieved_tensor = preprocess(retrieved_image).unsqueeze(0)
            retrieved_dino_emb = dino(retrieved_tensor)
            
            # Normalize DINO embeddings
            original_dino_emb = original_dino_emb / original_dino_emb.norm(dim=-1, keepdim=True)
            retrieved_dino_emb = retrieved_dino_emb / retrieved_dino_emb.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (no need to add 1 and divide by 2)
            soft_score = (original_dino_emb @ retrieved_dino_emb.T).item()
            all_soft_scores.append(soft_score)
    
    # Compute mean and standard deviation
    mean_soft_score = np.mean(all_soft_scores)
    std_soft_score = np.std(all_soft_scores)
    
    return mean_soft_score, std_soft_score


def compute_soft_retrieval_i2t(final_embedding1, final_embedding2, device="cuda", dataset='flickr8'):

    # Load Text Encoder model
    text_encoder = TextEncoder().to(device)
    text_encoder.eval()

    print(f"Computing soft retrieval scores for {dataset}")
    
    if dataset == 'flickr8' or dataset == 'flickr30':
        dataset = Flickr(
            root=f"../data/{dataset}/images",
            ann_file=f"../data/{dataset}/captions.txt",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    else:
        raise ValueError("Not supported dataset")
    
    # Convert to torch tensors and move to device
    final_embedding1 = torch.from_numpy(final_embedding1).float().to(device)
    final_embedding2 = torch.from_numpy(final_embedding2).float().to(device)

    # Normalize embeddings
    final_embedding1 = final_embedding1 / final_embedding1.norm(p=2, dim=-1, keepdim=True)
    final_embedding2 = final_embedding2 / final_embedding2.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity matrix and get top-1 indices
    cosine_similarity_matrix = final_embedding1 @ final_embedding2.T
    _, indices = torch.topk(cosine_similarity_matrix, largest=True, k=1)

    all_soft_scores = []
    
    with torch.no_grad():
        for image_idx in range(len(dataset)):
            # Get original text and its embedding
            original_text = dataset[image_idx][1]
            original_text_emb = text_encoder(original_text, device)
            
            # Get retrieved text and its embedding
            retrieved_idx = indices[image_idx].item()
            retrieved_text = dataset[retrieved_idx][1]
            retrieved_text_emb = text_encoder(retrieved_text, device)
            
            # Normalize embeddings
            original_text_emb = original_text_emb / original_text_emb.norm(dim=-1, keepdim=True)
            retrieved_text_emb = retrieved_text_emb / retrieved_text_emb.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (no need to add 1 and divide by 2)
            soft_score = (original_text_emb @ retrieved_text_emb.T).item()
            all_soft_scores.append(soft_score)
    
    # Compute mean and standard deviation
    mean_soft_score = np.mean(all_soft_scores)
    std_soft_score = np.std(all_soft_scores)
    
    return mean_soft_score, std_soft_score


def visualize_nearest_neighbors(dataset, image_embeddings, num_examples=5, k=5):

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import random
    
    # Normalize embeddings
    normalized_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
    
    # Convert to numpy for cosine similarity
    embeddings_np = normalized_embeddings.detach().cpu().numpy()
    
    # Get random indices for example images
    total_images = len(dataset)
    example_indices = random.sample(range(total_images), num_examples)
    
    for idx in example_indices:
        # Compute similarities
        query_embedding = embeddings_np[idx:idx+1]
        similarities = cosine_similarity(query_embedding, embeddings_np)[0]
        
        # Get top k+1 indices (including the query image)
        top_indices = np.argsort(similarities)[::-1][:k+1]
        
        # Create a figure with k+1 subplots
        fig, axes = plt.subplots(1, k+1, figsize=(15, 3))
        fig.suptitle(f'Query Image and its {k} Nearest Neighbors')
        
        # Plot query image
        query_img = dataset[idx]['image']
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query\n{dataset[idx]["label"]}')
        axes[0].axis('off')
        
        # Plot nearest neighbors
        for i, nn_idx in enumerate(top_indices[1:], 1):
            nn_img = dataset[int(nn_idx)]['image']
            axes[i].imshow(nn_img)
            axes[i].set_title(f'NN {i}\n{dataset[int(nn_idx)]["label"]}\nSim: {similarities[int(nn_idx)]:.3f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"query_image_{idx}.png")


def visualize_with_clip_score(embedding1, embedding2, mode='i2t', device: str = "cuda:1", dataset_name: str = 'flickr30', k: int = 5, num_queries: int = 30):
    # Input validation
    if mode not in ['i2t', 't2i']:
        raise ValueError("Mode must be either 'i2t' or 't2i'")
    if dataset_name not in ['flickr8', 'flickr30']:
        raise ValueError("Dataset must be either 'flickr8' or 'flickr30'")

    # Model setup
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Dataset setup
    dataset = Flickr(
        root=f"../data/{dataset_name}/images",
        ann_file=f"../data/{dataset_name}/captions.txt",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    # Prepare embeddings
    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    
    embedding1 = embedding1 / embedding1.norm(p=2, dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity matrix based on mode
    cosine_similarity_matrix = torch.matmul(embedding1, embedding2.T)

    # Get top k retrievals for the first num_queries
    _, indices = torch.topk(cosine_similarity_matrix[:num_queries], largest=True, k=k)
    
    results = []
    query_indices = range(min(num_queries, cosine_similarity_matrix.shape[0]))
    
    def process_image(image_tensor):
        """Helper function to process image tensor to PIL image"""
        # Convert to numpy and move channels to last dimension
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Scale from [0,1] to [0,255] and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Explicitly create RGB image
        return Image.fromarray(image_np, mode='RGB')

    def compute_clip_score(image_pil, caption, model, processor, device):
        """Helper function to compute CLIP score for an image-caption pair"""
        inputs = processor(
            images=image_pil,
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine = torch.matmul(image_features, text_features.T)
            clip_score = torch.max(torch.tensor(100.0) * cosine, torch.tensor(0.0))
            
        return clip_score

    for query_idx in query_indices:
        query_results = []
        
        if mode == 'i2t':
            # Image to text mode
            query_image = dataset[query_idx][0]
            query_caption = dataset[query_idx][1]
            query_image_pil = process_image(query_image)
            
            # Get original score
            original_score = compute_clip_score(query_image_pil, query_caption, model, processor, device)
            
            # Process each retrieved caption
            for retrieved_idx in indices[query_idx]:
                retrieved_caption = dataset[int(retrieved_idx.item())][1]
                retrieved_score = compute_clip_score(query_image_pil, retrieved_caption, model, processor, device)
                relative_score = retrieved_score / original_score
                
                query_results.append({
                    'query_type': 'image',
                    'query_image': query_image_pil,
                    'query_caption': query_caption,
                    'retrieved_type': 'text',
                    'retrieved_caption': retrieved_caption,
                    'relative_score': relative_score.item()
                })
                
        else:
            # Text to image mode
            query_caption = dataset[query_idx][1]
            query_image = dataset[query_idx][0]
            query_image_pil = process_image(query_image)
            
            # Get original score
            original_score = compute_clip_score(query_image_pil, query_caption, model, processor, device)
            
            # Process each retrieved image
            for retrieved_idx in indices[query_idx]:
                retrieved_image = dataset[int(retrieved_idx.item())][0]
                retrieved_image_pil = process_image(retrieved_image)
                retrieved_score = compute_clip_score(retrieved_image_pil, query_caption, model, processor, device)
                relative_score = retrieved_score / original_score
                
                query_results.append({
                    'query_type': 'text',
                    'query_caption': query_caption,
                    'query_image': query_image_pil,
                    'retrieved_type': 'image',
                    'retrieved_image': retrieved_image_pil,
                    'relative_score': relative_score.item()
                })
        
        results.append(query_results)
    
    def visualize_results(results, mode, k, dataset_name):
        """
        Visualize the retrieved results with their relative CLIP scores and save each query as a separate image
        """
        # Create the output directory if it doesn't exist
        output_dir = f'../data/{dataset_name}/qualitative_retrieval'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, query_results in enumerate(results):
            if mode == 'i2t':
                # Create figure with two subplots: one for image and one for text
                fig, (ax_image, ax_text) = plt.subplots(2, 1, figsize=(10, 12))
                
                # Add query image at the top
                query_image = query_results[0]['query_image']
                ax_image.imshow(query_image, alpha=1)
                ax_image.set_title('Query Image', pad=10)
                ax_image.axis('off')
                
                # Add retrieved captions below
                caption_text = "Retrieved Captions:\n\n"
                for j, result in enumerate(query_results):
                    caption_text += f"{j+1}. CLIP Score: {result['relative_score']:.2f}\n"
                    caption_text += f"   {result['retrieved_caption']}\n\n"
                
                ax_text.text(0.05, 0.95, caption_text,
                            va='top', ha='left',
                            wrap=True,
                            fontsize=12)
                ax_text.axis('off')
                
            else:  # t2i mode
                # Create figure with all subplots in one row
                fig, axes = plt.subplots(1, k, figsize=(15, 3))
                fig.suptitle(f"Query: {query_results[0]['query_caption']}", fontsize=14)
                
                # Create a grid of retrieved images
                for j, result in enumerate(query_results):
                    axes[j].imshow(result['retrieved_image'], alpha=1)
                    axes[j].set_title(f"CLIP Score: {result['relative_score']:.2f}")
                    axes[j].axis('off')
            
            plt.tight_layout()
            
            # Save the figure
            output_path = f'{output_dir}/results_{mode}_query_{i}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()


    # Visualize results
    visualize_results(results, mode, k, dataset_name)
    
    # Calculate and return overall statistics
    all_scores = [result['relative_score'] for query_result in results for result in query_result]
    return np.mean(all_scores), np.std(all_scores)


def save_checkpoint(trainer, path):
    """Save the complete training state in a single checkpoint file.
    
    Parameters
    ----------
    trainer : Trainer
        The trainer object containing all models and states
    path : str
        Path where to save the checkpoint file
    """
    checkpoint = {
        # SpectralNet1
        'spectralnet1_attributes': {k: v for k, v in trainer.spectralnet1.__dict__.items() 
                                  if not k.startswith('_') and k != 'spec_net' and k != 'device'},
        'spectralnet1_spec_net_state': trainer.spectralnet1.spec_net.state_dict() if hasattr(trainer.spectralnet1, 'spec_net') else None,
        'spectralnet1_orthonorm_weights': trainer.spectralnet1.spec_net.orthonorm_weights if hasattr(trainer.spectralnet1.spec_net, 'orthonorm_weights') else None,
        
        # SpectralNet2
        'spectralnet2_attributes': {k: v for k, v in trainer.spectralnet2.__dict__.items() 
                                  if not k.startswith('_') and k != 'spec_net' and k != 'device'},
        'spectralnet2_spec_net_state': trainer.spectralnet2.spec_net.state_dict() if hasattr(trainer.spectralnet2, 'spec_net') else None,
        'spectralnet2_orthonorm_weights': trainer.spectralnet2.spec_net.orthonorm_weights if hasattr(trainer.spectralnet2.spec_net, 'orthonorm_weights') else None,
        
        # CCA Projections
        'projection1': trainer.projection1 if hasattr(trainer, 'projection1') else None,
        'projection2': trainer.projection2 if hasattr(trainer, 'projection2') else None,
        
        # MMD Network
        'mmd_model_state': trainer.mmd_model.state_dict() if hasattr(trainer, 'mmd_model') else None,
        
        # Training flags
        'with_se': trainer.with_se if hasattr(trainer, 'with_se') else None,
        'with_cca': trainer.with_cca if hasattr(trainer, 'with_cca') else None,
        'with_mmd': trainer.with_mmd if hasattr(trainer, 'with_mmd') else None
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(trainer, path):
    """Load the complete training state from a single checkpoint file.
    
    Parameters
    ----------
    trainer : Trainer
        The trainer object to load the state into
    path : str
        Path to the checkpoint file
    """
    checkpoint = torch.load(path)
    
    # Load SpectralNet1
    for k, v in checkpoint['spectralnet1_attributes'].items():
        setattr(trainer.spectralnet1, k, v)
    
    if checkpoint['spectralnet1_spec_net_state'] is not None:
        trainer.spectralnet1.spec_net.load_state_dict(checkpoint['spectralnet1_spec_net_state'])
        if checkpoint['spectralnet1_orthonorm_weights'] is not None:
            trainer.spectralnet1.spec_net.orthonorm_weights = checkpoint['spectralnet1_orthonorm_weights']
    
    # Load SpectralNet2
    for k, v in checkpoint['spectralnet2_attributes'].items():
        setattr(trainer.spectralnet2, k, v)
    
    if checkpoint['spectralnet2_spec_net_state'] is not None:
        trainer.spectralnet2.spec_net.load_state_dict(checkpoint['spectralnet2_spec_net_state'])
        if checkpoint['spectralnet2_orthonorm_weights'] is not None:
            trainer.spectralnet2.spec_net.orthonorm_weights = checkpoint['spectralnet2_orthonorm_weights']
    
    # Load CCA Projections
    if checkpoint['projection1'] is not None:
        trainer.projection1 = checkpoint['projection1']
    if checkpoint['projection2'] is not None:
        trainer.projection2 = checkpoint['projection2']
    
    # Load MMD Network
    if checkpoint['mmd_model_state'] is not None:
        trainer.mmd_model.load_state_dict(checkpoint['mmd_model_state'])
    
    # Load training flags
    if checkpoint['with_se'] is not None:
        trainer.with_se = checkpoint['with_se']
    if checkpoint['with_cca'] is not None:
        trainer.with_cca = checkpoint['with_cca']
    if checkpoint['with_mmd'] is not None:
        trainer.with_mmd = checkpoint['with_mmd']


def visualize_office(embeddings1, embeddings2, labels1, labels2, num_samples=5, k=3):
    """
    Plot nearest neighbors between embeddings1 and embeddings2 based on cosine similarity.
    """
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms
    
    def get_nn_indices(source, target, k):
        # Normalize embeddings for cosine similarity
        source_norm = F.normalize(source, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(source_norm, target_norm.t())
        
        # Get top k similar indices
        _, indices = similarity.topk(k, dim=1)
        return indices
    
    # Randomly sample indices
    sample_indices = np.random.choice(len(embeddings1), num_samples, replace=False)
    
    # Plot settings
    fig, axs = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    # Plot embeddings1 -> embeddings2
    for idx, sample_idx in enumerate(sample_indices):
        nn_indices = get_nn_indices(embeddings1[sample_idx:sample_idx+1], embeddings2, k)[0]
        
        axs[0, idx].bar(range(k), 
                       F.cosine_similarity(embeddings1[sample_idx:sample_idx+1], 
                                        embeddings2[nn_indices]).cpu(),
                       color='blue')
        axs[0, idx].set_title(f'Domain1→2\nSample class: {labels1[sample_idx]}\nNN classes: {[labels2[i] for i in nn_indices]}')
        axs[0, idx].set_ylim(0, 1)
    
    # Plot embeddings2 -> embeddings1
    for idx, sample_idx in enumerate(sample_indices):
        nn_indices = get_nn_indices(embeddings2[sample_idx:sample_idx+1], embeddings1, k)[0]
        
        axs[1, idx].bar(range(k), 
                       F.cosine_similarity(embeddings2[sample_idx:sample_idx+1], 
                                        embeddings1[nn_indices]).cpu(),
                       color='red')
        axs[1, idx].set_title(f'Domain2→1\nSample class: {labels2[sample_idx]}\nNN classes: {[labels1[i] for i in nn_indices]}')
        axs[1, idx].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visual_results/office.png')


def calc_recall(similarity, labels=None, descending=True):
    n = similarity.shape[0]
    ranks = torch.zeros(n)
    top1 = torch.zeros(n)

    for index in range(n):
        inds = torch.argsort(similarity[index], descending=descending)
        rank = torch.where(inds == index)[0]
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * torch.sum(ranks < 1).float() / len(ranks)
    r5 = 100.0 * torch.sum(ranks < 5).float() / len(ranks)
    r10 = 100.0 * torch.sum(ranks < 10).float() / len(ranks)

    return r1, r5, r10