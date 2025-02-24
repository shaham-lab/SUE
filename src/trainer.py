import torch
import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from general_utils import calc_recall
from scipy.stats import pearsonr
from spectralnet import SpectralNet
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score


from mmd import *
from data import *
from encoders import *
from general_utils import *
from spectralnet._utils import *
    

class Trainer:
    def __init__(self, dataset_name='', n_parallel=500, n_eigenvectors=10, n_components=8, device=torch.device("cuda:1"), configs={}):
        self.device = device
        self.n_parallel = n_parallel
        self.n_components = n_components
        self.n_eigenvectors = n_eigenvectors
        self.dataset_name = dataset_name

        config1 = configs["spectralnets"][0]
        config2 = configs["spectralnets"][1]

        self.spectralnet1 = SpectralNet(**config1)
        self.spectralnet2 = SpectralNet(**config2)

        self.mmd_model = MMDNet(n_components).to(device)

    def fit(self, train_set, with_se=True, with_cca=True, with_mmd=True):
        self.with_se = with_se
        self.with_cca = with_cca
        self.with_mmd = with_mmd
        
        self.train_modality1 = train_set[0]
        self.train_modality2 = train_set[1]

        self.train_non_parallel1 = self.train_modality1[:-self.n_parallel]
        self.train_non_parallel2 = self.train_modality2[:-self.n_parallel]

        self.train_parallel1 = self.train_modality1[-self.n_parallel:]
        self.train_parallel2 = self.train_modality2[-self.n_parallel:]

        self.embeddings1 = self.train_modality1.detach().cpu().numpy()
        self.embeddings2 = self.train_modality2.detach().cpu().numpy()
        self.embeddings2 = self.embeddings2[: , :self.embeddings1.shape[1]]
        
        if with_se:
            self.spectralnet1.fit(self.train_modality1.float())
            self.spectralnet2.fit(self.train_modality2.float())

            self.spectralnet1.transform(self.train_modality1.float())
            self.spectralnet2.transform(self.train_modality2.float())

            self.embeddings1 = self.spectralnet1.embeddings_
            self.embeddings2 = self.spectralnet2.embeddings_
        
        if with_cca:
            print("Compute the CCA projections from the few parallel samples..")
            self.cca = CCA(n_components=self.n_components)
            self.cca.fit(
                self.embeddings1[-self.n_parallel :],
                self.embeddings2[-self.n_parallel :],
            )

            self.projection1 = self.cca.x_rotations_
            self.projection2 = self.cca.y_rotations_

            print("Project the entire modalities using the CCA projections..")
            self.embeddings1 = self.embeddings1 @ self.projection1
            self.embeddings2 = self.embeddings2 @ self.projection2

        if with_mmd:
            self.embeddings1, self.embeddings2, self.mmd_model = fine_tune_alignment_using_mmd_network(
                X=self.embeddings1,
                Y=self.embeddings2,
                device=self.device,
                epochs=100,
                batch_size=32,
                n_scales=3,
            )


    def test(self, test_set, visual=False):
        self.test_modality1 = test_set[0]
        self.test_modality2 = test_set[1]

        test_embeddings1 = self.test_modality1.detach().cpu().numpy()
        test_embeddings2 = self.test_modality2.detach().cpu().numpy()
        test_embeddings2 = test_embeddings2[: , :test_embeddings1.shape[1]]

        if self.with_se:
            self.spectralnet1.transform(self.test_modality1.float())
            self.spectralnet2.transform(self.test_modality2.float())

            test_embeddings1 = self.spectralnet1.embeddings_
            test_embeddings2 = self.spectralnet2.embeddings_

        if self.with_cca:
            test_embeddings1 = test_embeddings1 @ self.projection1
            test_embeddings2 = test_embeddings2 @ self.projection2

        if self.with_mmd:
            test_embeddings1 = self.mmd_model(torch.from_numpy(test_embeddings1).float().to(self.device))
            test_embeddings1 = test_embeddings1.detach().cpu().numpy()
            test_embeddings2 = test_embeddings2.astype(float)

        
        print("Compute Recall for test set..")
        self._compute_recall(test_embeddings1, test_embeddings2, type='text-to-image')
        self._compute_recall(test_embeddings1, test_embeddings2, type='image-to-text')

    
    def retrieve_images_for_caption(self, caption, test_set, k=5):
        text_encoder = TextEncoder()
        image_embeddings,_ = self._get_test_embeddings(test_set)

        with torch.no_grad():
            caption_embedding = text_encoder([caption], self.device)
        
        if self.with_se:
            self.spectralnet2.transform(caption_embedding.float())
            caption_embedding = self.spectralnet2.embeddings_
            
        if self.with_cca:
            caption_embedding = caption_embedding @ self.projection2
            
        
        similarities = np.dot(caption_embedding, image_embeddings.T)
        top_indices = np.argsort(similarities[0])[-k:][::-1]
        
        return top_indices
    
    
    def get_text_embeddings(self, texts):
        text_encoder = TextEncoder()
        with torch.no_grad():
            text_embeddings = text_encoder(texts, self.device)
        
        if self.with_se:
            self.spectralnet2.transform(text_embeddings.float())
            text_embeddings = self.spectralnet2.embeddings_
            
        if self.with_cca:
            text_embeddings = text_embeddings @ self.projection2
        
        return text_embeddings


    def get_image_embeddings(self, image):
        image_encoder = ImageEncoder().to(self.device)

        with torch.no_grad():
            image_embeddings = image_encoder(image, self.device)
        
        if self.with_se:
            self.spectralnet1.transform(image_embeddings.float())
            image_embeddings = self.spectralnet1.embeddings_
            
        if self.with_cca:
            image_embeddings = image_embeddings @ self.projection1

        if self.with_mmd:
            image_embeddings = self.mmd_model(torch.from_numpy(image_embeddings).float().to(self.device))
            image_embeddings = image_embeddings.detach().cpu().numpy()

        return image_embeddings


    def _get_test_embeddings(self, test_set):
        self.test_modality1 = test_set[0]
        self.test_modality2 = test_set[1]

        test_embeddings1 = self.test_modality1.detach().cpu().numpy()
        test_embeddings2 = self.test_modality2.detach().cpu().numpy()

        if self.with_se:
            self.spectralnet1.transform(self.test_modality1.float())
            self.spectralnet2.transform(self.test_modality2.float())

            test_embeddings1 = self.spectralnet1.embeddings_
            test_embeddings2 = self.spectralnet2.embeddings_

        if self.with_cca:
            test_embeddings1 = test_embeddings1 @ self.projection1
            test_embeddings2 = test_embeddings2 @ self.projection2

        if self.with_mmd:
            test_embeddings1 = self.mmd_model(torch.from_numpy(test_embeddings1).float().to(self.device))
            test_embeddings1 = test_embeddings1.detach().cpu().numpy()
            test_embeddings2 = test_embeddings2.astype(float)

        return test_embeddings1, test_embeddings2

    
    def _compute_recall(self, embedding1, embedding2, type='text-to-image'):
        embedding1 = embedding1 / np.linalg.norm(
            embedding1, axis=1, keepdims=True
        )
        embedding2 = embedding2 / np.linalg.norm(
            embedding2, axis=1, keepdims=True
        )

        if type == 'text-to-image':
            cosine_similarity_matrix = embedding2 @ embedding1.T
            recall_type_to_print = 'Text-to-image Recall: '
        
        elif type == 'image-to-text':
            cosine_similarity_matrix = embedding1 @ embedding2.T
            recall_type_to_print = 'Image-to-text Recall: '

        r1, r5, r10 = calc_recall(
            torch.from_numpy(cosine_similarity_matrix), labels=None
        )
        
        print(recall_type_to_print)
        print(f"Recall@1: {r1}")
        print(f"Recall@5: {r5}")
        print(f"Recall@10: {r10}")
        print("-"*20)
