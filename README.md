# VLM-Infer: Vision-Language Model Inference Pipeline

VLM-Infer is a flexible and extensible inference pipeline for Vision-Language Models (VLMs). It provides a unified interface for running inference with various VLMs, including LLaVA, Llama-Vision, and GPT-4V, while supporting different datasets and inference types.

## Features

- 🚀 Support for multiple VLM architectures:
  - LLaVA (7B and 13B)
  - Llama-Vision (11B and 70B)
  - GPT-4V (via OpenAI API)
- 📊 Multiple dataset formats:
  - VToM (Visual Theory of Mind)
  - COCO
  - MMBench
- 🛠 Flexible preprocessing pipeline
- 📈 Comprehensive analysis tools
- 🔄 Easy to extend with new models and datasets

## Installation

### Prerequisites

1. Python 3.8 or higher
2. CUDA-compatible GPU (for running large models)
3. API keys for model access:
   - HuggingFace token for LLaVA and Llama models
   - OpenAI API key for GPT-4V

### Environment Setup

1. Clone the repository:

```bash
bash
git clone https://github.com/itayitzhak/vlm_infer.git
cd vlm_infer
```

2. Create and activate a virtual environment (conda is recommended):

```bash
conda create -n vlm_infer python=3.10
conda activate vlm_infer
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package:

```bash
pip install -e .
```

5. Set up environment variables:

```bash
export HF_TOKEN=<your_huggingface_token>
export OPENAI_API_KEY=<your_openai_api_key>
```


## Quick Start

1. Prepare your configuration file:

```yaml
model:
  name: "llava-v1.5-7b"
  type: "hf"
  checkpoint: "llava-hf/llava-1.5-7b-hf"
  temperature: 0.0
  torch_dtype: "float16"
  device_map: "auto"
  is_chat_model: true
```

This is an example configuration file for the LLaVA 7B model. It specifies the model name, type, checkpoint, temperature, and dataset.

2. Run the inference pipeline:

```bash
python vlm_infer/main.py \
    --config vlm_infer/configs/vtom_llava1.5_7b_config.yaml \
    --dataset vtom # currently only vtom is supported
```

## Supported Models

### LLaVA Models
- `llava_7B`: LLaVA 1.5 7B model
- `llava_13B`: LLaVA 1.5 13B model

### Llama-Vision Models
- `llama3.2_11B`: Llama 3.2 11B Vision model
- `llama3.2_70B`: Llama 3.2 70B Vision model

### API Models
- `gpt4v`: GPT-4V through OpenAI API

## Inference Types

VLM-Infer supports various inference types:

1. `standard`: Basic image + question inference
2. `image_story`: Image + story + question
3. `story_only`: Story + question (no image)
4. `image_captions`: Image + captions + question
5. `captions_only`: Captions + question (no image)
6. `story_captions`: Story + captions + question

Select the inference type using the `--inference_type` argument or in your dataset config.

## Adding New Components

### Adding a New Dataset

1. Create a new dataset class in `vlm_infer/data/`:

```python
from .base import BaseDataset

class YourDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your dataset
        
    def _get_base_item(self, idx):
        # Implement item retrieval
        pass
```

2. Register your dataset in `vlm_infer/data/__init__.py`:

```python
DATASET_REGISTRY["your_dataset"] = YourDataset
```

### Adding a New Model

1. Create a new model class in appropriate location:

```python
from .base import BaseModel

class YourModel(BaseModel):
    def load(self):
        # Initialize your model
        pass
        
    def generate(self, inputs):
        # Implement generation logic
        pass
```

2. Register your model in `vlm_infer/models/__init__.py`:

```python
MODEL_REGISTRY["your_model"] = YourModel
```

## Analysis Tools

VLM-Infer includes comprehensive analysis tools for evaluating model outputs:

```bash
python -m vlm_infer.analysis.run_analysis \
    --results_dir results \
    --output_dir analysis_results \
    --save_plots
```

Analysis features include:
- F1 score computation
- Per-feature analysis
- Model comparison
- Visualization tools
- Story-based analysis

## Project Structure

```
vlm_infer/
├── data/               # Dataset implementations
├── models/            # Model implementations
├── preprocessing/     # Image and text preprocessing
├── inference/         # Inference pipeline
├── analysis/         # Analysis tools
├── configs/          # Configuration files
└── utils/            # Utility functions
```

## Configuration

The configuration system supports:
- Model-specific settings
- Dataset parameters
- Preprocessing options
- Inference settings

Example configurations are provided in the `configs/` directory.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
