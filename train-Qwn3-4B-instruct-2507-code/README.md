# Qwen3-4B Mental Health Multi-Task Fine-tuning

Fine-tune Qwen3-4B-Instruct-2507 on 6 mental health classification tasks using LoRA and log-probability evaluation.

## What it does
- **Multi-task learning**: One model handles stress detection, depression classification, and suicide risk assessment
- **LoRA fine-tuning**: Efficient parameter-efficient training
- **Log-probability evaluation**: Novel scoring method that compares log probabilities directly
- **8-bit quantization**: Memory-efficient training

## Tasks

The model is trained on 6 different mental health classification tasks:

1. **Stress Detection** (Binary): Predicts if a Reddit post expresses stress (1) or not (0)
2. **Depression Binary** (Binary): Predicts if a Reddit post shows depression (1) or no depression (0)
3. **Depression Severity** (4-class): Predicts depression level: Minimal (0), Mild (1), Moderate (2), Severe (3)
4. **Suicide Ideation** (Binary): Predicts if a post indicates suicidal thoughts (1) or not (0)
5. **Suicide Risk Binary** (Binary): Predicts if a user shows suicide risk indicators (1) or not (0)
6. **Suicide Risk Severity** (5-class): Predicts risk level: Supportive (1), Indicator (2), Ideation (3), Behavior (4), Attempt (5)

## Datasets

The framework uses 4 mental health datasets:

- **Dreaddit**: Reddit posts from stress-related subreddits (stress detection)
- **Reddit Depression Dataset**: Posts with depression severity labels (depression tasks)
- **SDCNL**: Posts from suicide-related subreddits (suicide ideation)
- **500 Reddit Users**: User-level posts with suicide risk labels (suicide risk tasks)

## Key Features

- **Multi-task learning**: One model handles 6 different mental health classification tasks
- **LoRA fine-tuning**: Much more efficient than full fine-tuning
- **Log-probability evaluation**: New method that directly compares log probabilities instead of using standard classification
- **8-bit quantization**: Saves memory during training
- **Multiple LoRA configs**: 9 different setups to experiment with
- **HPC support**: SLURM scripts for cluster computing

## Installation

You'll need:
- Python 3.8+
- CUDA 12.4+ (for GPU training)
- At least 16GB GPU memory (the 4B model is pretty big)

```bash
# Clone and setup
git clone <repository-url>
cd train-Qwn3-4B-instruct-2507-code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data cache/huggingface cache/nltk_data cache/wandb
```

## Data Setup

Put your CSV files in the `data/` folder:

```
data/
├── dreaddit_StressAnalysis - Sheet1.csv          # Stress data
├── Reddit_depression_dataset.csv                  # Depression data  
├── SDCNL.csv                                     # Suicide ideation data
└── 500_Reddit_users_posts_labels.csv             # Suicide risk data
```

Each CSV needs a text column and a label column. The code handles the label mapping automatically.

## Usage

### Basic Training

```python
from qwen3_lora_multitask_weighted_optimized import (
    Qwen3LoRAMultiTaskTrainer, MultiTaskConfig, create_optimized_task_configs
)

# Setup tasks and config
tasks = create_optimized_task_configs()
lora_config = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

config = MultiTaskConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    use_weighted_loss=True,
    output_dir="./qwen3_trained_model",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Train
trainer = Qwen3LoRAMultiTaskTrainer(config, tasks, lora_config)
trainer.train()
```

### With Log-Probability Evaluation

```python
from improved_logprob_implementation import create_improved_logprob_trainer

# Use the log-prob trainer instead
Qwen3LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Qwen3LoRAMultiTaskTrainer)

trainer = Qwen3LoRAMultiTaskTrainerWithLogProb(
    config, tasks, lora_config, 
    alpha=5.0,  # sigmoid sharpness
    beta=0.3    # CE vs BACC trade-off
)

trainer.train()

# Evaluate using log probabilities
metrics = trainer.evaluate_task_with_logprob(task_config, test_data)
```

### Running on HPC

```bash
# Set your project directory
export PROJECT_DIR="/path/to/your/project"

# Submit the job
sbatch qwen3_config1_logprob.slurm
```

## Configuration Options

### LoRA Settings

There are 9 different LoRA configurations you can try:

| Config | r | alpha | dropout | target_modules |
|--------|---|-------|---------|----------------|
| 1 | 8 | 16 | 0.1 | q_proj, k_proj, v_proj, o_proj |
| 2 | 16 | 32 | 0.1 | q_proj, k_proj, v_proj, o_proj |
| 3 | 8 | 16 | 0.1 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj |
| 4 | 8 | 16 | 0.2 | q_proj, k_proj, v_proj, o_proj |
| 5 | 4 | 32 | 0.1 | q_proj, k_proj, v_proj, o_proj |
| 6 | 8 | 16 | 0.1 | All modules |
| 7 | 16 | 16 | 0.05 | q_proj, k_proj, v_proj, o_proj |
| 8 | 12 | 24 | 0.15 | q_proj, k_proj, v_proj, o_proj, gate_proj |
| 9 | 6 | 12 | 0.1 | q_proj, k_proj, v_proj |

### Log-Probability Parameters

- **alpha**: Controls sigmoid sharpness (1-20, default: 5.0)
- **beta**: Balances cross-entropy vs BACC loss (0.1-0.5, default: 0.3)

### Training Settings

```python
MultiTaskConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    max_length=512,
    batch_size=4,
    learning_rate=5e-4,
    num_epochs=3,
    warmup_steps=100,
    use_8bit=True,  # Saves memory
    use_weighted_loss=True
)
```

## Evaluation

The code tracks several metrics:

- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Handles class imbalance better
- **F1-Score**: Macro and weighted F1 scores
- **Log-Probability Scoring**: Direct probability comparison method

## Advanced Features

### BACC Surrogate Loss

The code implements a balanced accuracy surrogate loss:

```
LBACC = 1 - (1 / Σγc) * Σγc * TPRc
```

Where TPRc is the true positive rate for class c, and γc are class weights. The final loss combines this with cross-entropy: L = LCE + β * LBACC

### Log-Probability Scoring

Instead of standard classification, this method:
- Calculates log p(0|prompt) vs log p(1|prompt)
- Picks the label with higher log-probability
- Works with multi-token labels

## Results

Results are saved in JSON format:

```json
{
  "config_1_logprob": {
    "lora_config": {...},
    "improved_logprob_params": {
      "alpha": 5.0,
      "beta": 0.3
    },
    "task_results": {
      "task1_stress": {
        "improved_logprob": {
          "accuracy": 0.85,
          "balanced_accuracy": 0.82,
          "f1_macro": 0.81,
          "f1_weighted": 0.84
        }
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use 8-bit quantization

2. **Path Issues**
   - Set `PROJECT_DIR` environment variable
   - Make sure data files are in the right place
   - Check cache directory permissions

3. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check your Python path
   - Make sure virtual environment is activated

### Memory Optimization

```python
# Enable 8-bit quantization
config = MultiTaskConfig(use_8bit=True)

# Reduce batch size
config.batch_size = 2

# Use gradient checkpointing
training_args.gradient_checkpointing = True
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen3-4B-Instruct-2507 model by Alibaba Cloud
- Hugging Face Transformers library
- PEFT library for parameter-efficient fine-tuning
- Mental health datasets used in this research

## References

- [Qwen3 Model Documentation](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This framework is designed for research purposes. Please ensure compliance with data privacy regulations and ethical guidelines when working with mental health data.
