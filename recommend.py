import argparse
import numpy as np
from embeddings import EmbeddingAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Generate music recommendations using audio embeddings.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings parquet file')
    parser.add_argument('--listens', type=str, required=False, help='Path to listens parquet file (optional)')
    parser.add_argument('--user_history', type=str, required=True, help='Comma-separated list of item_ids representing user history')
    parser.add_argument('--n_recommendations', type=int, default=10, help='Number of recommendations to return')
    args = parser.parse_args()

    # Parse user history
    user_history = [x.strip() for x in args.user_history.split(',') if x.strip()]

    # Try to convert to int if possible
    try:
        user_history = [int(x) for x in user_history]
    except ValueError:
        pass  # Keep as string if not convertible

    analyzer = EmbeddingAnalyzer(args.embeddings, args.listens)
    analyzer.load_data()

    rec_features = analyzer.compute_recommendation_features(user_history, n_recommendations=args.n_recommendations)

    if rec_features:
        print("Recommended item_ids:")
        for item, score in zip(rec_features['recommended_items'], rec_features['similarity_scores']):
            print(f"{item}\tScore: {score:.4f}")
        print(f"\nRecommendation diversity: {rec_features['recommendation_diversity']:.3f}")
        print(f"Mean novelty: {rec_features['mean_novelty']:.3f}")
    else:
        print("No recommendations could be generated for the provided user history.")


if __name__ == "__main__":
    main() 