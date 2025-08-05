# Hakka ASR

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Format](#data-format)
- [Usage](#usage)
- [Parameters](#parameters)
- [Directory Structure](#directory-structure)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/YuChiChen12/Hakka_ASR.git
```

2. **Create virtual environment**

```bash
# Windows
python -m venv hakka_asr_env
hakka_asr_env\Scripts\activate

# Linux/Mac
python3 -m venv hakka_asr_env
source hakka_asr_env/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify CUDA installation (for GPU training)**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## Data Format

### CSV Structure

Your dataset must be a CSV file with the following columns:

| Column       | Type   | Description                                  | Example                        |
| ------------ | ------ | -------------------------------------------- | ------------------------------ |
| `audio_path` | string | Absolute or relative path to audio file      | `/path/to/audio/sample001.wav` |
| `客語漢字`   | string | Hakka text in traditional Chinese characters | `這係一句客語`                 |
| `客語拼音`   | string | Hakka text in pinyin notation                | `li he chid kiu hag ngi`       |

### Example CSV (hakka_data.csv)

```csv
audio_path,客語漢字,客語拼音
/data/audio/sample001.wav,這係一句客語,li he chid kiu hag ngi
/data/audio/sample002.wav,你好嗎,ngi ho ma
/data/audio/sample003.wav,食飯了無,sii fen liau mo
/data/audio/sample004.wav,今晡日天氣真好,kiun bu ngid tien hi chin ho
/data/audio/sample005.wav,我愛客家話,ngai oi hag ka fa
```

## Usage

### Basic Training Commands

#### Track 1: 客語漢字 (Chinese Characters with WER)

```bash
python train_hakka_asr_final.py \
    --csv_path hakka_data.csv \
    --track track1 \
    --model_name openai/whisper-small \
    --fp16 \
    --use_tensorboard
```

#### Track 2: 客語拼音 (Hakka Pinyin with SER)

```bash
python train_hakka_asr_final.py \
    --csv_path hakka_data.csv \
    --track track2 \
    --model_name openai/whisper-small \
    --fp16 \
    --use_tensorboard
```

### Advanced Training Examples

#### High-Quality Training (Large Model)

```bash
python train_hakka_asr_final.py \
    --csv_path hakka_data.csv \
    --track track1 \
    --model_name openai/whisper-large-v3 \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_steps 10000 \
    --train_split 0.9 \
    --fp16 \
    --use_tensorboard
```

#### Memory-Constrained Training

```bash
python train_hakka_asr_final.py \
    --csv_path hakka_data.csv \
    --track track1 \
    --model_name openai/whisper-small \
    --batch_size 2 \
    --grad_accumulation_steps 8 \
    --fp16 \
    --max_steps 5000
```

#### Fast Prototyping (Small Model)

```bash
python train_hakka_asr_final.py \
    --csv_path hakka_data.csv \
    --track track1 \
    --model_name openai/whisper-tiny \
    --batch_size 32 \
    --max_steps 2000 \
    --eval_steps 200
```

## Directory Structure

### Before Training

```
Hakka_ASR/
├── train_hakka_asr_final.py      # Main training script
├── inference.py                   # Inference script (to be created)
├── requirements_final.txt         # Dependencies
├── hakka_data.csv                # Your training data
└── audio/                        # Audio files directory
    ├── sample001.wav
    ├── sample002.wav
    └── ...
```

### After Training

```
Hakka_ASR/
├── hakka_asr_models/             # Generated models directory
│   ├── track1/                   # Track 1 models (漢字)
│   │   ├── checkpoint-1000/      # Training checkpoints
│   │   ├── checkpoint-2000/
│   │   ├── final_model/          # Final trained model
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── preprocessor_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── ...
│   │   └── runs/                 # TensorBoard logs
│   └── track2/                   # Track 2 models (拼音)
│       └── ...
├── hakka_data_train_track1.csv   # Generated training split
├── hakka_data_val_track1.csv     # Generated validation split
├── hakka_asr_training.log        # Training logs
└── ...
```

### Key Files Explanation

- **`final_model/`**: Complete trained model ready for inference
- **`checkpoint-*/`**: Training checkpoints for resuming
- **`runs/`**: TensorBoard logs for monitoring training
- **`*_train_*.csv`**: Automatically generated training splits
- **`hakka_asr_training.log`**: Detailed training logs

## Performance Optimization

### GPU Optimization

#### Memory Management

```bash
# For limited GPU memory
--batch_size 2 --grad_accumulation_steps 8 --fp16

# For medium GPU memory
--batch_size 8 --grad_accumulation_steps 2 --fp16

# For high-end GPU memory
--batch_size 16 --grad_accumulation_steps 1 --fp16
```

#### Training Speed

- **Enable FP16**: Always use `--fp16` for 2x speed improvement
- **Optimal batch size**: Find largest batch size that fits in GPU memory
- **Gradient accumulation**: Use to simulate larger batch sizes
- **Model size**: Balance between accuracy and training time

### CPU Optimization

- **Data workers**: Increase `--num_workers` for faster data loading
- **Memory**: Close unnecessary applications
- **Storage**: Use SSD for faster data access

### Hyperparameter Tuning

#### Learning Rate Guidelines

- **Large models**: `5e-6` to `1e-5`
- **Small models**: `1e-5` to `2e-5`
- **Fine-tuning rule**: 40x smaller than pre-training LR

#### Training Steps

- **Small dataset (<1 hour)**: 2000-5000 steps
- **Medium dataset (1-8 hours)**: 5000-15000 steps
- **Large dataset (8+ hours)**: 10000-30000 steps

#### Batch Size Strategy

```python
# Effective batch size = batch_size × grad_accumulation_steps
# Aim for effective batch size of 16-32 for best results

# Option 1: Large batch, no accumulation
--batch_size 16 --grad_accumulation_steps 1

# Option 2: Small batch, high accumulation (for limited memory)
--batch_size 4 --grad_accumulation_steps 4

# Option 3: Medium batch, medium accumulation
--batch_size 8 --grad_accumulation_steps 2
```

### Monitoring and Debugging

#### TensorBoard Monitoring

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir ./hakka_asr_models/track1/runs

# Open browser to: http://localhost:6006
```

#### Log Analysis

```bash
# Monitor training progress
tail -f hakka_asr_training.log

# Check for errors
grep -i error hakka_asr_training.log

# Monitor GPU usage
nvidia-smi -l 1
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 2 --grad_accumulation_steps 8

# Enable FP16
--fp16

# Use smaller model
--model_name openai/whisper-small
```

#### Slow Training

```bash
# Enable FP16
--fp16

# Increase batch size (if memory allows)
--batch_size 32

# Reduce evaluation frequency
--eval_steps 2000
```

#### Poor Performance

```bash
# Increase training steps
--max_steps 10000

# Use larger model
--model_name openai/whisper-large-v3

# Adjust learning rate
--learning_rate 5e-6
```
