# Hakka ASR

## Environment Setup

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/YuChiChen12/Hakka_ASR.git
```

2. **Create virtual environment & Install dependencies**

```bash
ml load miniconda3/24.11.1   # or use another method to activate the conda env
conda create -n hakka_asr python=3.10
pip install -r requirements.txt
```

3. **Verify CUDA installation (for GPU training)**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## Data Format

### CSV Structure
The data preprocessing script is only for data at [GitLab repo](https://speech.nchc.org.tw/).
The output CSV file with the following columns:

audio_path,客語漢字,客語拼音,備註
data/dataset_2/熱身賽_錄製語料_大埔腔_4H/DM203J2056/DM203J2056_001.wav,五月節愛食粽,ng31 ngied54 zied21 oi33 shid54 zung53,
data/dataset_2/熱身賽_錄製語料_大埔腔_4H/DM203J2056/DM203J2056_002.wav,寒著愛戴嘴揞,hon113 do31 oi33 dai53 zhoi53 em33,
data/dataset_2/熱身賽_錄製語料_大埔腔_4H/DM203J2056/DM203J2056_003.wav,你哪位毋鬆爽,hn113 ne53 vui53 m113 sung33 song31,

## Usage

### Basic Training Commands

```bash
sbatch data_preprocessing.sb
sbatch train.sb
sbatch test.sb
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
