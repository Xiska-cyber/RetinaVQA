# RetinaVQA

## Causal Graph-Guided Vision-Language Model for OCT Screening

### Overview

RetinaVQA is a deep learning system for automated diabetic retinopathy screening from OCT images.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
from retinavqa.eval.inference import RetinaVQAPredictor

predictor = RetinaVQAPredictor(
    "retinavqa/models/best_model.pt",
    "retinavqa/models/causal_graph.pt"
)

result = predictor.predict("image.jpg")
print(result["prediction"])
```

### Results

| Metric | Value |
|--------|-------|
| Accuracy | 95.8% |
| AUC | 0.992 |
| Sensitivity | 96.8% |
| Specificity | 93.8% |
