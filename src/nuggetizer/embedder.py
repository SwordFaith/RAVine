''' Embedding representation model for the clustering phase of nuggetization '''

from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self,
                 model_name,
                 max_seq_length: int=512):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.cuda()
        self.model.eval()
    
    def encode(self, sentences, batch_size: int=32):
        with torch.no_grad():
            embeddings = self.model.encode(sentences,
                                           show_progress_bar=True,
                                           batch_size=batch_size,
                                           convert_to_tensor=False, 
                                           max_length=self.max_seq_length,
                                           normalize_embeddings=True)
        return embeddings
