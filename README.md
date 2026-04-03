# RetinaVQA

## Causal Graph-Guided Vision-Language Model for OCT Screening

### Overview

RetinaVQA is a deep learning system for automated diabetic retinopathy screening from OCT images.

### Installation
## Model Weights Download

The model files are hosted on Google Drive due to size constraints:

- **best_model.pt** (49 MB): [Download from Google Drive](https://drive.google.com/file/d/1ZHoZMTc60-6l7ylkP3z3SUj4yuzU2t1S/view?usp=sharing)
- **causal_graph.pt**: [Download from Google Drive](https://drive.google.com/file/d/1B0tIGIISML8J8g23YJX8xbSRowFsQnsU/view?usp=sharing)

After downloading, place both files in: `retinavqa/models/`


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
