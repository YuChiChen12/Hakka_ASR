"""
Enhanced Hakka ASR Training with SpecAugment Integration
Improved version based on your updated train.py with additional features
"""

import os
import random
import argparse
import logging
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import json
import nltk
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from tqdm import tqdm
import sys

# Import SpecAugment integration
try:
    from hakka_specaugment import HakkaSpecAugment, create_hakka_spec_augment
except ImportError:
    print("Warning: SpecAugment not found. Training will proceed without augmentation.")
    HakkaSpecAugment = None

# Import Noise Augmentation integration
try:
    from hakka_noise_augmentation import (
        HakkaNoisePipeline, 
        CombinedHakkaAugmentation, 
        create_hakka_noise_pipeline
    )
except ImportError:
    print("Warning: Noise augmentation not found. Training will proceed without noise augmentation.")
    HakkaNoisePipeline = None
    CombinedHakkaAugmentation = None

# Distributed training logging control
if int(os.environ.get("LOCAL_RANK", 0)) != 0:
    logging.getLogger().setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_hakka_asr_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedHakkaASRDataset(Dataset):
    """
    Enhanced Hakka ASR Dataset with SpecAugment support and improved flexibility
    """
    
    def __init__(self, data, processor, track="track1", from_dataframe=False, 
                 use_spec_augment=False, use_noise_augment=False, dialect="general", 
                 spec_augment_prob=0.8, noise_augment_prob=0.6, noise_dir=None):
        """
        Initialize dataset with enhanced features
        
        Args:
            data: DataFrame or CSV path
            processor: Whisper processor
            track: "track1" (漢字) or "track2" (拼音)
            from_dataframe: Whether data is DataFrame or CSV path
            use_spec_augment: Enable SpecAugment
            use_noise_augment: Enable noise augmentation
            dialect: Dialect for SpecAugment tuning
            spec_augment_prob: Probability of applying SpecAugment
            noise_augment_prob: Probability of applying noise augmentation
            noise_dir: Directory containing background noise files
        """
        if from_dataframe:
            self.df = data.reset_index(drop=True)
        else:
            self.df = pd.read_csv(data)
        
        self.processor = processor
        self.track = track
        self.use_spec_augment = use_spec_augment and HakkaSpecAugment is not None
        self.use_noise_augment = use_noise_augment and HakkaNoisePipeline is not None
        self.spec_augment_prob = spec_augment_prob
        self.noise_augment_prob = noise_augment_prob
        self.text_column = "客語漢字" if track == "track1" else "客語拼音"
        
        # Initialize SpecAugment if enabled
        if self.use_spec_augment:
            self.spec_augment = create_hakka_spec_augment(dialect, track)
            logger.info(f"SpecAugment enabled for {dialect} dialect, {track}")
        else:
            self.spec_augment = None
            if HakkaSpecAugment is None:
                logger.info("SpecAugment not available - continuing without augmentation")
            else:
                logger.info("SpecAugment disabled")
        
        # Initialize Noise Augmentation if enabled
        if self.use_noise_augment:
            self.noise_augment = create_hakka_noise_pipeline(
                dialect=dialect, 
                conservative=True,  # Conservative for tonal language
                noise_dir=noise_dir
            )
            logger.info(f"Noise augmentation enabled for {dialect} dialect")
            logger.info(f"Noise types: {self.noise_augment.noise_types}")
            logger.info(f"SNR range: {self.noise_augment.snr_range}dB")
        else:
            self.noise_augment = None
            if HakkaNoisePipeline is None:
                logger.info("Noise augmentation not available")
            else:
                logger.info("Noise augmentation disabled")
        
        # Initialize combined augmentation if both are enabled
        if self.use_spec_augment and self.use_noise_augment and CombinedHakkaAugmentation is not None:
            self.combined_augment = CombinedHakkaAugmentation(
                noise_augment=self.noise_augment,
                spec_augment=self.spec_augment,
                noise_first=True,  # Research-based: noise before spectral augmentation
                combined_prob=min(spec_augment_prob, noise_augment_prob)
            )
            logger.info("Combined augmentation pipeline enabled (Noise→SpecAugment)")
        else:
            self.combined_augment = None
        
        logger.info(f"Loaded {len(self.df)} samples for {track}")
        logger.info(f"Target column: {self.text_column}")
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset integrity"""
        missing_audio = self.df['audio_path'].isna().sum()
        missing_text = self.df[self.text_column].isna().sum()
        
        if missing_audio > 0:
            logger.warning(f"Found {missing_audio} samples with missing audio paths")
        if missing_text > 0:
            logger.warning(f"Found {missing_text} samples with missing {self.text_column}")
        
        # Check if audio files exist (sample check)
        sample_files = self.df['audio_path'].dropna().head(5).tolist()
        missing_files = [f for f in sample_files if not Path(f).exists()]
        if missing_files:
            logger.warning(f"Sample audio files not found: {missing_files[:3]}")
        else:
            logger.info("Sample audio file validation passed")
    
    def set_training_mode(self, training: bool):
        """Set training mode for SpecAugment"""
        if self.spec_augment:
            if training:
                self.spec_augment.train()
            else:
                self.spec_augment.eval()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Return a single sample with optional SpecAugment"""
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        text = str(row[self.text_column])
        
        # Load and preprocess audio
        try:
            audio_array, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio_array.shape[0] > 1:
                audio_array = torch.mean(audio_array, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_array = resampler(audio_array)
            
            audio_array = audio_array.squeeze()
            
            # Apply noise augmentation if enabled (time-domain augmentation)
            if (self.use_noise_augment and 
                self.noise_augment is not None and 
                self.spec_augment is not None and 
                self.spec_augment.training and 
                random.random() < self.noise_augment_prob):
                
                audio_array = self.noise_augment(audio_array)
            
            # Convert to numpy for processor
            audio_array = audio_array.numpy() if isinstance(audio_array, torch.Tensor) else audio_array
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if audio loading fails
            audio_array = np.zeros(16000)
        
        # Process audio using processor
        try:
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.squeeze(0)  # [n_mels, time_frames]
            
            # Apply SpecAugment if enabled and in training mode
            if (self.use_spec_augment and 
                self.spec_augment is not None and 
                self.spec_augment.training and 
                random.random() < self.spec_augment_prob):
                
                input_features = self.spec_augment(input_features)
                
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            # Return dummy features if processing fails
            input_features = torch.zeros((80, 300))  # Standard mel spec shape
        
        return {
            "input_features": input_features,
            "text": text,
            "audio_path": audio_path,  # For debugging
        }


@dataclass
class EnhancedDataCollatorSpeechSeq2SeqWithPadding:
    """
    Enhanced data collator with better error handling and logging
    """
    processor: Any
    decoder_start_token_id: int
    max_length: int = 448

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Combine features into a batch with robust error handling"""
        try:
            # Handle audio inputs
            input_features = [{"input_features": feature["input_features"].cpu().numpy()} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

            # Handle text labels
            texts = [f["text"] for f in features]
            labels_batch = self.processor.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_length
            )

            # Replace padding tokens with -100
            labels = labels_batch["input_ids"]
            if "attention_mask" in labels_batch:
                labels = labels.masked_fill(labels_batch["attention_mask"].ne(1), -100)
            else:
                # If no attention mask, assume all tokens are valid except padding (pad_token_id)
                pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
                labels = labels.masked_fill(labels.eq(pad_token_id), -100)

            # Remove BOS token if present
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch
            
        except Exception as e:
            logger.error(f"Error in data collator: {e}")
            # Log problematic files for debugging
            audio_paths = [f.get("audio_path", "unknown") for f in features]
            logger.error(f"Problematic audio files: {audio_paths}")
            raise


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def character_error_rate(reference, hypothesis):
    """Calculate Character Error Rate"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edit_distance = nltk.edit_distance(ref_chars, hyp_chars)
    len_ref = len(ref_chars)
    if len_ref == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return edit_distance / len_ref


def word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_syl = reference.split()
    hyp_syl = hypothesis.split()
    edit_distance = nltk.edit_distance(ref_syl, hyp_syl)
    len_ref = len(ref_syl)
    if len_ref == 0:
        return 0.0 if len(hyp_syl) == 0 else 1.0
    return edit_distance / len_ref


def setup_model_and_processor(args):
    """Initialize Whisper model and processor with enhanced error handling"""
    try:
        model_dir = Path(args.model_name)
        if model_dir.exists() and model_dir.is_dir():
            # Local fine-tuned model
            logger.info("Loading fine-tuned model from local directory...")
            processor = WhisperProcessor.from_pretrained(
                model_dir.as_posix(), 
                language=args.language, 
                task=args.task
            )
            model = WhisperForConditionalGeneration.from_pretrained(model_dir.as_posix())
        else:
            # HuggingFace Hub model with subfolder support
            logger.info("Loading pre-trained model from HuggingFace Hub...")
            repo_id = args.model_name.split("/")[0] + "/" + args.model_name.split("/")[1] if args.model_name.count("/") >= 1 else args.model_name
            subfolder = "/".join(args.model_name.split("/")[2:]) if args.model_name.count("/") >= 2 else None

            if subfolder:
                logger.info(f"Loading from repo: {repo_id}, subfolder: {subfolder}")
                processor = WhisperProcessor.from_pretrained(repo_id, subfolder=subfolder)
                model = WhisperForConditionalGeneration.from_pretrained(repo_id, subfolder=subfolder)
            else:
                processor = WhisperProcessor.from_pretrained(repo_id)
                model = WhisperForConditionalGeneration.from_pretrained(repo_id)
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to default Whisper model...")
        try:
            processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            logger.info("Successfully loaded fallback model")
        except Exception as fallback_e:
            logger.error(f"Fallback model loading failed: {fallback_e}")
            raise

    # Configure model
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    
    logger.info(f"Model and processor loaded successfully")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, processor


def create_enhanced_datasets(csv_path, processor, track, train_split=0.8, seed=42, 
                           use_spec_augment=False, use_noise_augment=False, dialect="general", 
                           spec_augment_prob=0.8, noise_augment_prob=0.6, noise_dir=None):
    """Create enhanced datasets with combined augmentation support"""
    logger.info(f"Creating enhanced datasets from {csv_path}")
    logger.info(f"SpecAugment: {'enabled' if use_spec_augment else 'disabled'}")
    logger.info(f"Noise augmentation: {'enabled' if use_noise_augment else 'disabled'}")
    logger.info(f"Dialect: {dialect}")
    logger.info(f"SpecAugment probability: {spec_augment_prob}")
    logger.info(f"Noise augmentation probability: {noise_augment_prob}")
    if noise_dir:
        logger.info(f"Noise directory: {noise_dir}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Total samples: {len(df)}")
        
        # Validate required columns
        required_cols = ['audio_path', '客語漢字', '客語拼音']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create train/validation split
        train_df = df.sample(frac=train_split, random_state=seed)
        val_df = df.drop(train_df.index)
        
        logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Create datasets
        logger.info("Creating train dataset with augmentation...")
        train_dataset = EnhancedHakkaASRDataset(
            train_df, processor, track, from_dataframe=True, 
            use_spec_augment=use_spec_augment, use_noise_augment=use_noise_augment,
            dialect=dialect, spec_augment_prob=spec_augment_prob,
            noise_augment_prob=noise_augment_prob, noise_dir=noise_dir
        )
        train_dataset.set_training_mode(True)
        
        logger.info("Creating validation dataset...")
        val_dataset = EnhancedHakkaASRDataset(
            val_df, processor, track, from_dataframe=True, 
            use_spec_augment=False, use_noise_augment=False, 
            dialect=dialect  # No augmentation for validation
        )
        val_dataset.set_training_mode(False)
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def train_model(args):
    """Enhanced training function with comprehensive error handling"""
    logger.info(f"Starting enhanced training for {args.track}")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"SpecAugment: {args.use_spec_augment}")
    logger.info(f"Noise augmentation: {args.use_noise_augment}")
    logger.info(f"Dialect: {args.dialect}")
    if hasattr(args, 'noise_dir') and args.noise_dir:
        logger.info(f"Noise directory: {args.noise_dir}")
    
    # Setup model and processor
    model, processor = setup_model_and_processor(args)
    
    # Create datasets
    train_dataset, val_dataset = create_enhanced_datasets(
        args.csv_path, processor, args.track, args.train_split, args.seed,
        args.use_spec_augment, args.use_noise_augment, args.dialect, 
        args.spec_augment_prob, args.noise_augment_prob, args.noise_dir
    )
    
    # Data collator
    data_collator = EnhancedDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=args.max_length
    )
    
    # Metrics computation
    def compute_metrics_wrapper(pred):
        try:
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            if args.track == "track1":
                total_edits = sum(nltk.edit_distance(list(ref), list(hyp))
                                for ref, hyp in zip(label_str, pred_str))
                total_ref = sum(len(ref) for ref in label_str) or 1
                cer = total_edits / total_ref
                logger.info(f"Corpus-level CER: {cer*100:.2f}%")
                return {"cer": cer}
            else:
                total_edits = sum(nltk.edit_distance(ref.split(), hyp.split())
                                for ref, hyp in zip(label_str, pred_str))
                total_ref = sum(len(ref.split()) for ref in label_str) or 1
                wer = total_edits / total_ref
                logger.info(f"Corpus-level WER: {wer*100:.2f}%")
                return {"wer": wer}
        except Exception as e:
            logger.error(f"Error in metrics computation: {e}")
            return {"cer" if args.track == "track1" else "wer": 1.0}
    
    # Training arguments - determine output suffix based on augmentations
    suffix_parts = []
    if args.use_spec_augment:
        suffix_parts.append("specaugment")
    if hasattr(args, 'use_noise_augment') and args.use_noise_augment:
        suffix_parts.append("noise")
    
    if suffix_parts:
        output_suffix = "_" + "_".join(suffix_parts)
    else:
        output_suffix = "_baseline"
        
    output_dir = f"{args.output_dir}/{args.track}{output_suffix}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=args.fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"] if args.use_tensorboard else [],
        load_best_model_at_end=True,
        metric_for_best_model="cer" if args.track == "track1" else "wer",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        # Seq2Seq specific parameters  
        # DDP settings (only needed for multi-GPU)
        predict_with_generate=True,
        generation_max_length=args.max_length,
        logging_first_step=True,
        logging_strategy="steps",
        seed=args.seed,
        # Enhanced settings
        save_safetensors=True,
        push_to_hub=False,
        hub_model_id=None,
        hub_strategy="end",
    )
    
    logger.info("Training arguments configured:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        processing_class=processor,
    )
    
    # Start training
    logger.info("Starting enhanced training...")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model
        final_model_path = f"{output_dir}/final_model"
        logger.info(f"Saving final model to {final_model_path}")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        
        logger.info(f"Enhanced training completed! Model saved to {final_model_path}")
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Enhanced Hakka ASR with SpecAugment Integration")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data/hakka_data.csv",
                       help="Path to CSV file")
    parser.add_argument("--track", type=str, choices=["track1", "track2"], required=True,
                       help="Track 1: 客語漢字 (CER), Track 2: 客語拼音 (WER)")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                       help="Whisper model name")
    parser.add_argument("--language", type=str, default="Chinese",
                       help="Target language for fine-tuning")
    parser.add_argument("--task", type=str, default="transcribe",
                       choices=["transcribe", "translate"], help="Task type")
    parser.add_argument("--output_dir", type=str, default="./enhanced_hakka_asr_models",
                       help="Output directory for models")
    
    # SpecAugment arguments
    parser.add_argument("--use_spec_augment", action="store_true",
                       help="Enable SpecAugment")
    parser.add_argument("--dialect", type=str, choices=["general", "dapu", "zhaoan"],
                       default="general", help="Hakka dialect for parameter tuning")
    parser.add_argument("--spec_augment_prob", type=float, default=0.8,
                       help="Probability of applying SpecAugment")
    
    # Noise Augmentation arguments
    parser.add_argument("--use_noise_augment", action="store_true",
                       help="Enable noise augmentation")
    parser.add_argument("--noise_augment_prob", type=float, default=0.6,
                       help="Probability of applying noise augmentation")
    parser.add_argument("--noise_dir", type=str, default=None,
                       help="Directory containing background noise files")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging steps")
    parser.add_argument("--max_length", type=int, default=448, help="Maximum sequence length")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard")
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    
    # Setup logging
    logger.info("=" * 70)
    logger.info("ENHANCED HAKKA ASR WITH SPECAUGMENT INTEGRATION")
    logger.info("=" * 70)
    
    # Download NLTK data
    logger.info("Downloading NLTK data...")
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK data already available")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
        logger.info("NLTK data downloaded successfully")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {args.output_dir}")
    
    # Log all arguments
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 70)
    
    # Run training
    try:
        trainer = train_model(args)
        logger.info("Enhanced training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()