"""
YAMBDA Audio Embeddings Analysis
Analyze audio embeddings to understand music characteristics and their relationship
with organic vs algorithmic interactions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EmbeddingAnalyzer:
    """Analyze audio embeddings for recommendation insights"""
    
    def __init__(self, embeddings_path, listens_path=None):
        self.embeddings_path = embeddings_path
        self.listens_path = listens_path
        self.embeddings_df = None
        self.listens_df = None
        self.embedding_matrix = None
        
    def load_data(self):
        """Load embeddings and optionally listening data"""
        print("Loading embeddings data...")
        self.embeddings_df = pd.read_parquet(self.embeddings_path)
        print(f"Loaded embeddings for {len(self.embeddings_df):,} tracks")
        
        # Extract embedding vectors
        if 'embed' in self.embeddings_df.columns:
            # Convert list of embeddings to numpy array
            self.embedding_matrix = np.vstack(self.embeddings_df['embed'].values)
            print(f"Embedding dimensions: {self.embedding_matrix.shape}")
        
        if self.listens_path:
            print("\nLoading listening data for context...")
            self.listens_df = pd.read_parquet(self.listens_path)
            print(f"Loaded {len(self.listens_df):,} listening events")
            
        return self.embeddings_df
    
    def basic_statistics(self):
        """Compute basic statistics of embeddings"""
        stats = {
            'n_tracks': len(self.embeddings_df),
            'embedding_dim': self.embedding_matrix.shape[1],
            'mean_norm': np.mean(np.linalg.norm(self.embedding_matrix, axis=1)),
            'std_norm': np.std(np.linalg.norm(self.embedding_matrix, axis=1)),
            'sparsity': np.mean(self.embedding_matrix == 0)
        }
        
        # Component-wise statistics
        stats['component_means'] = np.mean(self.embedding_matrix, axis=0)
        stats['component_stds'] = np.std(self.embedding_matrix, axis=0)
        
        return stats
    
    def analyze_embedding_space(self):
        """Analyze the structure of the embedding space"""
        print("\nAnalyzing embedding space structure...")
        
        # Sample for computational efficiency
        n_sample = min(10000, len(self.embedding_matrix))
        sample_indices = np.random.choice(len(self.embedding_matrix), n_sample, replace=False)
        sample_embeddings = self.embedding_matrix[sample_indices]
        
        # Compute pairwise distances
        distances = pdist(sample_embeddings, metric='cosine')
        
        analysis = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'distance_percentiles': np.percentile(distances, [10, 25, 50, 75, 90])
        }
        
        # Analyze nearest neighbors
        distance_matrix = squareform(distances)
        nn_distances = np.sort(distance_matrix, axis=1)[:, 1:6]  # 5 nearest neighbors
        analysis['mean_nn_distance'] = np.mean(nn_distances)
        
        return analysis, distances
    
    def cluster_embeddings(self, n_clusters=50, method='kmeans'):
        """Cluster embeddings to find music groups"""
        print(f"\nClustering embeddings using {method}...")
        
        # Sample for efficiency
        n_sample = min(20000, len(self.embedding_matrix))
        sample_indices = np.random.choice(len(self.embedding_matrix), n_sample, replace=False)
        sample_embeddings = self.embedding_matrix[sample_indices]
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(sample_embeddings)
            
            # Compute cluster quality metrics
            silhouette = silhouette_score(sample_embeddings, cluster_labels, sample_size=5000)
            inertia = clusterer.inertia_
            
            # Find cluster centers
            centers = clusterer.cluster_centers_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
            cluster_labels = clusterer.fit_predict(sample_embeddings)
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            silhouette = silhouette_score(sample_embeddings, cluster_labels) if n_clusters_found > 1 else -1
            centers = None
            
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        
        return {
            'labels': cluster_labels,
            'cluster_sizes': cluster_sizes,
            'silhouette_score': silhouette,
            'centers': centers,
            'sample_indices': sample_indices
        }
    
    def analyze_popularity_bias(self):
        """Analyze how embedding characteristics relate to popularity"""
        if self.listens_df is None:
            print("Listening data not loaded. Skipping popularity analysis.")
            return None
        
        print("\nAnalyzing popularity bias in embeddings...")
        
        # Calculate track popularity
        track_popularity = self.listens_df.groupby('item_id').size()
        
        # Merge with embeddings
        embeddings_with_pop = self.embeddings_df.copy()
        embeddings_with_pop['popularity'] = embeddings_with_pop['item_id'].map(track_popularity).fillna(0)
        
        # Divide into popularity tiers
        # Handle zero popularity tracks separately
        embeddings_with_pop_nonzero = embeddings_with_pop[embeddings_with_pop['popularity'] > 0].copy()
        
        if len(embeddings_with_pop_nonzero) > 0:
            embeddings_with_pop_nonzero['popularity_tier'] = pd.qcut(
                embeddings_with_pop_nonzero['popularity'], 
                q=[0, 0.5, 0.8, 0.95, 1.0], 
                labels=['long_tail', 'medium', 'popular', 'very_popular'],
                duplicates='drop'
            )
            
            # Add zero popularity tracks as 'unplayed'
            embeddings_with_pop.loc[embeddings_with_pop['popularity'] == 0, 'popularity_tier'] = 'unplayed'
            embeddings_with_pop.loc[embeddings_with_pop['popularity'] > 0, 'popularity_tier'] = embeddings_with_pop_nonzero['popularity_tier']
        else:
            embeddings_with_pop['popularity_tier'] = 'unplayed'
        
        # Analyze embedding characteristics by tier
        tier_analysis = {}
        for tier in ['unplayed', 'long_tail', 'medium', 'popular', 'very_popular']:
            tier_mask = embeddings_with_pop['popularity_tier'] == tier
            if tier_mask.sum() > 0:
                tier_embeddings = self.embedding_matrix[tier_mask]
                
                # Compute diversity within tier
                if len(tier_embeddings) > 100:
                    sample_idx = np.random.choice(len(tier_embeddings), 100, replace=False)
                    tier_sample = tier_embeddings[sample_idx]
                    tier_distances = pdist(tier_sample, metric='cosine')
                    
                    tier_analysis[tier] = {
                        'count': tier_mask.sum(),
                        'mean_distance': np.mean(tier_distances),
                        'std_distance': np.std(tier_distances),
                        'embedding_variance': np.mean(np.var(tier_embeddings, axis=0))
                    }
                elif len(tier_embeddings) > 0:
                    # For small tiers, use all embeddings
                    tier_analysis[tier] = {
                        'count': tier_mask.sum(),
                        'mean_distance': 0 if len(tier_embeddings) < 2 else np.mean(pdist(tier_embeddings, metric='cosine')),
                        'std_distance': 0,
                        'embedding_variance': np.mean(np.var(tier_embeddings, axis=0))
                    }
        
        return tier_analysis, embeddings_with_pop
    
    def analyze_organic_vs_algo_preferences(self):
        """Analyze embedding patterns for organic vs algorithmic interactions"""
        if self.listens_df is None:
            print("Listening data not loaded. Skipping organic/algo analysis.")
            return None
        
        print("\nAnalyzing embedding patterns for organic vs algorithmic interactions...")
        
        # Get organic and algorithmic interaction item sets
        organic_items = self.listens_df[self.listens_df['is_organic'] == 1]['item_id'].unique()
        algo_items = self.listens_df[self.listens_df['is_organic'] == 0]['item_id'].unique()
        
        # Get embeddings for each set
        organic_mask = self.embeddings_df['item_id'].isin(organic_items)
        algo_mask = self.embeddings_df['item_id'].isin(algo_items)
        
        organic_embeddings = self.embedding_matrix[organic_mask]
        algo_embeddings = self.embedding_matrix[algo_mask]
        
        print(f"Organic items: {len(organic_embeddings):,}")
        print(f"Algorithmic items: {len(algo_embeddings):,}")
        
        # Compare distributions
        analysis = {
            'organic_mean_embedding': np.mean(organic_embeddings, axis=0),
            'algo_mean_embedding': np.mean(algo_embeddings, axis=0),
            'organic_std': np.mean(np.std(organic_embeddings, axis=0)),
            'algo_std': np.mean(np.std(algo_embeddings, axis=0))
        }
        
        # Compute centroid distance
        analysis['centroid_distance'] = cosine_similarity(
            analysis['organic_mean_embedding'].reshape(1, -1),
            analysis['algo_mean_embedding'].reshape(1, -1)
        )[0, 0]
        
        # Sample for diversity comparison
        n_sample = min(1000, len(organic_embeddings), len(algo_embeddings))
        organic_sample = organic_embeddings[np.random.choice(len(organic_embeddings), n_sample, replace=False)]
        algo_sample = algo_embeddings[np.random.choice(len(algo_embeddings), n_sample, replace=False)]
        
        # Compute intra-set diversity
        organic_diversity = np.mean(pdist(organic_sample, metric='cosine'))
        algo_diversity = np.mean(pdist(algo_sample, metric='cosine'))
        
        analysis['organic_diversity'] = organic_diversity
        analysis['algo_diversity'] = algo_diversity
        analysis['diversity_ratio'] = organic_diversity / algo_diversity
        
        return analysis
    
    def find_embedding_patterns(self):
        """Find interesting patterns in embeddings using PCA"""
        print("\nFinding embedding patterns with PCA...")
        
        # Fit PCA
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(self.embedding_matrix)
        
        # Analyze principal components
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Find number of components for 90% variance
        n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
        
        patterns = {
            'explained_variance': explained_var[:10],
            'cumulative_variance': cumulative_var[:10],
            'n_components_90': n_components_90,
            'top_components': pca.components_[:5]  # Top 5 principal components
        }
        
        return patterns, pca_embeddings
    
    def compute_recommendation_features(self, user_history, n_recommendations=10):
        """
        Compute content-based recommendation features based on user history
        This demonstrates how embeddings can be used for recommendations
        """
        print("\nComputing recommendation features...")
        
        # Get embeddings for user history
        history_mask = self.embeddings_df['item_id'].isin(user_history)
        if history_mask.sum() == 0:
            print("No embeddings found for user history")
            return None
        
        history_embeddings = self.embedding_matrix[history_mask]
        
        # Compute user profile as weighted average (recent items weighted more)
        weights = np.exp(np.linspace(-1, 0, len(history_embeddings)))
        weights = weights / weights.sum()
        user_profile = np.average(history_embeddings, axis=0, weights=weights)
        
        # Compute similarities to all items
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.embedding_matrix)[0]
        
        # Get top recommendations (excluding history)
        history_indices = np.where(history_mask)[0]
        similarities[history_indices] = -1  # Exclude already seen items
        
        top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
        top_items = self.embeddings_df.iloc[top_indices]['item_id'].values
        top_scores = similarities[top_indices]
        
        # Analyze recommendation characteristics
        rec_embeddings = self.embedding_matrix[top_indices]
        rec_diversity = np.mean(pdist(rec_embeddings, metric='cosine'))
        
        # Novelty: average distance from history items
        novelty_scores = []
        for rec_emb in rec_embeddings:
            distances = [cosine_similarity(rec_emb.reshape(1, -1), 
                                         hist_emb.reshape(1, -1))[0, 0] 
                        for hist_emb in history_embeddings]
            novelty_scores.append(1 - np.mean(distances))
        
        return {
            'user_profile': user_profile,
            'recommended_items': top_items,
            'similarity_scores': top_scores,
            'recommendation_diversity': rec_diversity,
            'novelty_scores': np.array(novelty_scores),
            'mean_novelty': np.mean(novelty_scores)
        }
    
    def visualize_embedding_space(self, method='tsne', n_samples=5000):
        """Visualize the embedding space"""
        print(f"\nVisualizing embedding space with {method.upper()}...")
        
        # Sample embeddings
        sample_idx = np.random.choice(len(self.embedding_matrix), 
                                     min(n_samples, len(self.embedding_matrix)), 
                                     replace=False)
        sample_embeddings = self.embedding_matrix[sample_idx]
        
        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(sample_embeddings)
        
        # If we have listening data, color by popularity or organic/algo
        colors = None
        if self.listens_df is not None:
            # Calculate item popularity for coloring
            item_popularity = self.listens_df.groupby('item_id').size()
            sample_items = self.embeddings_df.iloc[sample_idx]['item_id']
            colors = sample_items.map(item_popularity).fillna(0).values
            colors = np.log1p(colors)  # Log scale for better visualization
        
        return reduced_embeddings, colors, sample_idx

def create_embedding_visualizations(analyzer, results):
    """Create comprehensive visualizations for embedding analysis"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Embedding space visualization
    ax1 = plt.subplot(3, 3, 1)
    reduced_embeddings, colors, _ = analyzer.visualize_embedding_space(n_samples=3000)
    
    if colors is not None:
        scatter = ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            c=colors, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax1, label='Log Popularity')
    else:
        ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                   alpha=0.6, s=10)
    
    ax1.set_title('Embedding Space Visualization (t-SNE)')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    
    # 2. Distance distribution
    ax2 = plt.subplot(3, 3, 2)
    _, distances = analyzer.analyze_embedding_space()
    ax2.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Pairwise Distance Distribution')
    ax2.set_xlabel('Cosine Distance')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.3f}')
    ax2.legend()
    
    # 3. PCA explained variance
    ax3 = plt.subplot(3, 3, 3)
    patterns, _ = analyzer.find_embedding_patterns()
    ax3.plot(range(1, 11), patterns['explained_variance'], marker='o')
    ax3.set_title('PCA Explained Variance')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cluster sizes
    ax4 = plt.subplot(3, 3, 4)
    cluster_results = analyzer.cluster_embeddings(n_clusters=20)
    cluster_sizes = cluster_results['cluster_sizes'].values[:20]
    ax4.bar(range(len(cluster_sizes)), cluster_sizes)
    ax4.set_title(f'Cluster Sizes (Silhouette: {cluster_results["silhouette_score"]:.3f})')
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Number of Tracks')
    
    # 5. Popularity tier analysis
    ax5 = plt.subplot(3, 3, 5)
    tier_analysis, _ = analyzer.analyze_popularity_bias()
    if tier_analysis:
        # Sort tiers by logical order
        tier_order = ['unplayed', 'long_tail', 'medium', 'popular', 'very_popular']
        tiers = [t for t in tier_order if t in tier_analysis]
        diversities = [tier_analysis[t]['mean_distance'] for t in tiers]
        counts = [tier_analysis[t]['count'] for t in tiers]
        
        # Create bar plot
        bars = ax5.bar(tiers, diversities, alpha=0.7)
        ax5.set_title('Embedding Diversity by Popularity Tier')
        ax5.set_xlabel('Popularity Tier')
        ax5.set_ylabel('Mean Intra-tier Distance')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
    
    # 6. Organic vs Algorithmic patterns
    ax6 = plt.subplot(3, 3, 6)
    org_algo_analysis = analyzer.analyze_organic_vs_algo_preferences()
    if org_algo_analysis:
        categories = ['Organic', 'Algorithmic']
        diversities = [org_algo_analysis['organic_diversity'], 
                      org_algo_analysis['algo_diversity']]
        ax6.bar(categories, diversities, alpha=0.7, color=['green', 'blue'])
        ax6.set_title('Embedding Diversity: Organic vs Algorithmic')
        ax6.set_ylabel('Mean Pairwise Distance')
        
        # Add text annotation
        ax6.text(0.5, max(diversities) * 0.9, 
                f"Centroid Similarity: {org_algo_analysis['centroid_distance']:.3f}",
                ha='center', transform=ax6.transAxes)
    
    # 7. Component means comparison
    ax7 = plt.subplot(3, 3, 7)
    if org_algo_analysis:
        component_diff = (org_algo_analysis['organic_mean_embedding'] - 
                         org_algo_analysis['algo_mean_embedding'])[:50]
        ax7.bar(range(len(component_diff)), component_diff, alpha=0.7)
        ax7.set_title('Embedding Component Differences (Organic - Algorithmic)')
        ax7.set_xlabel('Component Index')
        ax7.set_ylabel('Mean Difference')
        ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 8. Embedding norms distribution
    ax8 = plt.subplot(3, 3, 8)
    norms = np.linalg.norm(analyzer.embedding_matrix, axis=1)
    ax8.hist(norms, bins=50, alpha=0.7, edgecolor='black')
    ax8.set_title('Embedding Norm Distribution')
    ax8.set_xlabel('L2 Norm')
    ax8.set_ylabel('Frequency')
    ax8.axvline(np.mean(norms), color='red', linestyle='--', 
                label=f'Mean: {np.mean(norms):.3f}')
    ax8.legend()
    
    # 9. Recommendation example
    ax9 = plt.subplot(3, 3, 9)
    # Create a sample user history
    sample_history = np.random.choice(analyzer.embeddings_df['item_id'].values, 10)
    rec_features = analyzer.compute_recommendation_features(sample_history)
    
    if rec_features:
        metrics = ['Similarity', 'Novelty']
        values = [np.mean(rec_features['similarity_scores']), 
                 rec_features['mean_novelty']]
        ax9.bar(metrics, values, alpha=0.7)
        ax9.set_title('Sample Recommendation Metrics')
        ax9.set_ylabel('Score')
        ax9.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function for embedding analysis"""
    print("YAMBDA Audio Embeddings Analysis")
    print("=" * 60)
    
    # Update these paths to match your setup
    data_path = "/Users/ananthgs/Downloads/code/yambda_retrieval_code"  # Update this
    embeddings_path = f"{data_path}/embeddings.parquet"
    listens_path = f"{data_path}/flat-50m/listens.parquet"  # Check if this exists
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(embeddings_path, listens_path)
    
    # Load data
    print("\n1. Loading Data...")
    analyzer.load_data()
    
    # Basic statistics
    print("\n2. Computing Basic Statistics...")
    stats = analyzer.basic_statistics()
    print(f"Number of tracks: {stats['n_tracks']:,}")
    print(f"Embedding dimensions: {stats['embedding_dim']}")
    print(f"Mean embedding norm: {stats['mean_norm']:.3f}")
    print(f"Sparsity: {stats['sparsity']:.3%}")
    
    # Analyze embedding space
    print("\n3. Analyzing Embedding Space...")
    space_analysis, _ = analyzer.analyze_embedding_space()
    print(f"Mean pairwise distance: {space_analysis['mean_distance']:.3f}")
    print(f"Distance range: [{space_analysis['min_distance']:.3f}, {space_analysis['max_distance']:.3f}]")
    print(f"Mean nearest neighbor distance: {space_analysis['mean_nn_distance']:.3f}")
    
    # Clustering
    print("\n4. Clustering Analysis...")
    cluster_results = analyzer.cluster_embeddings(n_clusters=20)
    print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")
    print(f"Largest cluster size: {cluster_results['cluster_sizes'].max()}")
    print(f"Smallest cluster size: {cluster_results['cluster_sizes'].min()}")
    
    # Popularity bias
    print("\n5. Analyzing Popularity Bias...")
    tier_analysis, embeddings_with_pop = analyzer.analyze_popularity_bias()
    if tier_analysis:
        for tier, stats in tier_analysis.items():
            print(f"\n{tier.upper()}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Mean diversity: {stats['mean_distance']:.3f}")
            print(f"  Embedding variance: {stats['embedding_variance']:.6f}")
    
    # Organic vs Algorithmic
    print("\n6. Analyzing Organic vs Algorithmic Patterns...")
    org_algo = analyzer.analyze_organic_vs_algo_preferences()
    if org_algo:
        print(f"Organic diversity: {org_algo['organic_diversity']:.3f}")
        print(f"Algorithmic diversity: {org_algo['algo_diversity']:.3f}")
        print(f"Diversity ratio (organic/algo): {org_algo['diversity_ratio']:.3f}")
        print(f"Centroid similarity: {org_algo['centroid_distance']:.3f}")
    
    # Find patterns
    print("\n7. Finding Embedding Patterns...")
    patterns, pca_embeddings = analyzer.find_embedding_patterns()
    print(f"Components needed for 90% variance: {patterns['n_components_90']}")
    print(f"First 5 components explain: {sum(patterns['explained_variance'][:5]):.1%} of variance")
    
    # Create visualizations
    print("\n8. Creating Visualizations...")
    create_embedding_visualizations(analyzer, {
        'stats': stats,
        'space_analysis': space_analysis,
        'cluster_results': cluster_results,
        'tier_analysis': tier_analysis,
        'org_algo': org_algo,
        'patterns': patterns
    })
    
    print("Statistics dictionary - fetch",stats.items())
    stats = analyzer.basic_statistics()

    # Save results
    results_df = pd.DataFrame([{
        'n_tracks': stats['n_tracks'],
        'embedding_dim': stats['embedding_dim'],
        'mean_distance': space_analysis['mean_distance'],
        'clustering_silhouette': cluster_results['silhouette_score'],
        'organic_diversity': org_algo['organic_diversity'] if org_algo else None,
        'algo_diversity': org_algo['algo_diversity'] if org_algo else None,
        'centroid_similarity': org_algo['centroid_distance'] if org_algo else None,
        'n_components_90': patterns['n_components_90']
    }])
    
    results_df.to_csv('embedding_analysis_results.csv', index=False)
    print("\nResults saved to 'embedding_analysis_results.csv'")
    
    print("\n" + "=" * 60)
    print("Embedding analysis complete!")
    
    # Demonstrate recommendation usage
    print("\n9. Recommendation Feature Example:")
    print("Creating content-based recommendations for a sample user...")
    
    # Get some popular organic items as sample history
    if analyzer.listens_df is not None:
        organic_popular = analyzer.listens_df[
            analyzer.listens_df['is_organic'] == 1
        ]['item_id'].value_counts().head(20).index.tolist()
        
        # Only use items that have embeddings
        available_items = set(analyzer.embeddings_df['item_id'].values)
        organic_popular = [item for item in organic_popular if item in available_items]
        
        if len(organic_popular) >= 5:
            sample_history = np.random.choice(organic_popular, 5)
            rec_features = analyzer.compute_recommendation_features(sample_history, n_recommendations=10)
            
            if rec_features:
                print(f"\nUser history: {len(sample_history)} items")
                print(f"Recommendation diversity: {rec_features['recommendation_diversity']:.3f}")
                print(f"Mean novelty: {rec_features['mean_novelty']:.3f}")
                print(f"Top recommendation similarity: {rec_features['similarity_scores'][0]:.3f}")
        else:
            print("Not enough organic popular items with embeddings for demo")

if __name__ == "__main__":
    main()
