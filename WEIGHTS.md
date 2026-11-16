# Model Weights

Real weights (`model.safetensors`, ~467 MB, 116.7M parameters) are hosted on Hugging Face:

https://huggingface.co/ayazelrico/Milyonus-Optic-Tracking-2.4

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ayazelrico/Milyonus-Optic-Tracking-2.4",
    filename="model.safetensors"
)
```

Existing browser algorithms and structure are untouched.
