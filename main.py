import argparse
import numpy as np
import pandas as pd
from embeddings import EmbeddingAnalyzer
from visualization import create_embedding_visualizations


def main():
    parser = argparse.ArgumentParser(description="YAMBDA Audio Embeddings Analysis Pipeline")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings parquet file')
    parser.add_argument('--listens', type=str, required=False, help='Path to listens parquet file (optional)')
    args = parser.parse_args()

    print("YAMBDA Audio Embeddings Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(args.embeddings, args.listens)

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
        for tier, tstats in tier_analysis.items():
            print(f"\n{tier.upper()}:")
            print(f"  Count: {tstats['count']:,}")
            print(f"  Mean diversity: {tstats['mean_distance']:.3f}")
            print(f"  Embedding variance: {tstats['embedding_variance']:.6f}")

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

if __name__ == "__main__":
    main() 