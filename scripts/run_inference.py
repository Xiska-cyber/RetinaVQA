"""
Main inference script for RetinaVQA
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retinavqa.eval.inference import RetinaVQAPredictor


def main():
    parser = argparse.ArgumentParser(description='RetinaVQA Inference')
    parser.add_argument('--model', type=str, 
                        default='retinavqa/models/best_model.pt',
                        help='Path to model weights (.pt)')
    parser.add_argument('--graph', type=str,
                        default='retinavqa/models/causal_graph.pt',
                        help='Path to causal graph (.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize predictor
    print("Loading RetinaVQA model...")
    predictor = RetinaVQAPredictor(args.model, args.graph, args.device)

    # Get all images
    input_path = Path(args.input)
    if input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
    else:
        images = [input_path]

    print(f"\nProcessing {len(images)} images...")
    print("-" * 40)

    # Run inference
    results = predictor.predict_batch(images)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print summary
    valid_results = [r for r in results if 'prediction' in r]
    if valid_results:
        abnormal_count = sum(1 for r in valid_results if r['prediction'] == 'Abnormal')
        print(f"\nSummary:")
        print(f"  Total images: {len(valid_results)}")
        print(f"  Abnormal: {abnormal_count}")
        print(f"  Normal: {len(valid_results) - abnormal_count}")


if __name__ == "__main__":
    main()
