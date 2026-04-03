"""
Inference utilities for RetinaVQA
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Dict


class RetinaVQAPredictor:
    """
    RetinaVQA Inference Wrapper

    Example:
        predictor = RetinaVQAPredictor(model_path, graph_path)
        result = predictor.predict("image.jpg")
        print(result['severity'], result['prediction'])
    """

    def __init__(self, model_path: str, graph_path: str, device: str = 'cuda'):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model weights
            graph_path: Path to causal graph file
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Import here to avoid circular imports
        from retinavqa.models.model import load_retinavqa

        print(f"Loading model from {model_path}...")
        self.model, self.edge_index, self.edge_weights = load_retinavqa(
            model_path, graph_path, self.device
        )
        print("Model loaded successfully!")

        self.transform = self._get_transform()

    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict severity for a single image

        Args:
            image_path: Path to OCT image

        Returns:
            Dictionary with 'severity' and 'prediction' keys
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            severity, uncertainty, _ = self.model(
                img_tensor, self.edge_index, self.edge_weights
            )

        severity_val = severity.item()

        return {
            'severity': severity_val,
            'prediction': 'Abnormal' if severity_val > 0.5 else 'Normal',
            'uncertainty': uncertainty.item() if hasattr(uncertainty, 'item') else 0,
            'image': str(image_path)
        }

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict severity for multiple images

        Args:
            image_paths: List of paths to OCT images

        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
                print(f"  {path.name}: {result['prediction']} (severity={result['severity']:.3f})")
            except Exception as e:
                print(f"  Error with {path.name}: {e}")
                results.append({'image': str(path), 'error': str(e)})
        return results
