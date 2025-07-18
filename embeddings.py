import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

warnings.filterwarnings('ignore')

class EmbeddingAnalyzer:
    """Analyze audio embeddings for recommendation insights"""
    # ... existing code ...
    def __init__(self, embeddings_path, listens_path=None):
        # ... existing code ...
    def load_data(self):
        # ... existing code ...
    def basic_statistics(self):
        # ... existing code ...
    def analyze_embedding_space(self):
        # ... existing code ...
    def cluster_embeddings(self, n_clusters=50, method='kmeans'):
        # ... existing code ...
    def analyze_popularity_bias(self):
        # ... existing code ...
    def analyze_organic_vs_algo_preferences(self):
        # ... existing code ...
    def find_embedding_patterns(self):
        # ... existing code ...
    def compute_recommendation_features(self, user_history, n_recommendations=10):
        # ... existing code ...
    def visualize_embedding_space(self, method='tsne', n_samples=5000):
        # ... existing code ... 