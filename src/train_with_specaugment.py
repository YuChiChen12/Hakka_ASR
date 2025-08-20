"""
Enhanced Hakka ASR Training with Integrated SpecAugment
Based on original train.py with SpecAugment integration
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
from typing import Any, Dict, List, Union
from tqdm import tqdm

# Import our custom SpecAugment integration
from hakka_specaugment import HakkaSpecAugment, create_hakka_spec_augment


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hakka_asr_specaugment_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HakkaASRDatasetWithSpecAugment(Dataset):
    """
    Enhanced HakkaASRDataset with integrated SpecAugment support
    """
    
    def __init__(self, csv_path, processor, track="track1", use_spec_augment=True, dialect="general"):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.track = track
        self.use_spec_augment = use_spec_augment
        self.dialect = dialect
        
        # Determine target column based on track
        self.text_column = "客語漢字" if track == "track1" else "客語拼音"
        
        # Initialize SpecAugment if enabled
        if self.use_spec_augment:
            self.spec_augment = create_hakka_spec_augment(dialect, track)
            logger.info(f"SpecAugment enabled for {dialect} dialect, {track}")
        else:
            self.spec_augment = None
            logger.info("SpecAugment disabled")
        
        logger.info(f"Loaded {len(self.df)} samples for {track}")
        logger.info(f"Target column: {self.text_column}")
    
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
        """
        Return a single sample with optional SpecAugment applied
        """
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
            
            audio_array = audio_array.squeeze().numpy()
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if audio loading fails
            audio_array = np.zeros(16000)
        
        # Process audio using processor to get input_features (mel spectrogram)
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.squeeze(0)  # [n_mels, time_frames]

        # Apply SpecAugment to the mel spectrogram if enabled and in training mode
        if (self.use_spec_augment and 
            self.spec_augment is not None and 
            self.spec_augment.training):
            
            # input_features shape: [n_mels, time_frames] 
            input_features = self.spec_augment(input_features)

        # Process text labels using tokenizer
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=448,
            padding=False,
        ).input_ids.squeeze(0)

        return {
            "input_features": input_features,  # torch.FloatTensor [n_mels, time_frames] 
            "labels": labels,                  # torch.LongTensor  [seq_len,]
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Enhanced data collator - unchanged from original as SpecAugment is applied at dataset level
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Handle audio inputs and text labels separately
        input_features = [{"input_features": feature["input_features"].cpu().numpy()} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

        # Handle text labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)

        # Replace padding tokens with -100 so they are ignored during loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def character_error_rate(reference, hypothesis):
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edit_distance = nltk.edit_distance(ref_chars, hyp_chars)
    len_ref = len(ref_chars)
    if len_ref == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return edit_distance / len_ref


def word_error_rate(reference, hypothesis):
    ref_syl = reference.split()
    hyp_syl = hypothesis.split()
    edit_distance = nltk.edit_distance(ref_syl, hyp_syl)
    len_ref = len(ref_syl)
    if len_ref == 0:
        return 0.0 if len(hyp_syl) == 0 else 1.0
    return edit_distance / len_ref


def setup_model_and_processor(args):
    """Initialize Whisper model and processor"""
    try:
        model_dir = Path(args.model_name)
        if model_dir.exists() and model_dir.is_dir():
            logger.info("Loading fine-tuned model from local directory...")
            processor = WhisperProcessor.from_pretrained(model_dir.as_posix(), language=args.language, task=args.task)
            model = WhisperForConditionalGeneration.from_pretrained(model_dir.as_posix())
        else:
            logger.info("Loading pre-trained model from HuggingFace Hub...")
            processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
            model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    
    logger.info(f"Model and processor loaded successfully")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, processor


def create_datasets(csv_path, processor, track, use_existing_split=True, train_split=0.8, seed=42, 
                   use_spec_augment=True, dialect="general"):
    """Create train/validation datasets with SpecAugment integration"""
    logger.info(f"Creating datasets from {csv_path}")
    logger.info(f"SpecAugment: {'enabled' if use_spec_augment else 'disabled'}")
    logger.info(f"Using existing split: {use_existing_split}")

    # Check if pre-split files exist from data_prepare.py
    csv_path_obj = Path(csv_path)
    train_csv = csv_path_obj.parent / f"{csv_path_obj.stem}_train{csv_path_obj.suffix}"
    test_csv = csv_path_obj.parent / f"{csv_path_obj.stem}_test{csv_path_obj.suffix}"
    
    if use_existing_split and train_csv.exists() and test_csv.exists():
        # Use existing split from data_prepare.py
        logger.info(f"Using existing train/test split from data_prepare.py:")
        logger.info(f"  Train CSV: {train_csv}")
        logger.info(f"  Test CSV: {test_csv}")
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(test_csv)
        
        logger.info(f"Loaded - Train samples: {len(train_df)}, Test samples: {len(val_df)}")
        
        # Save track-specific versions for compatibility
        track_train_csv = str(train_csv).replace('.csv', f'_{track}.csv')
        track_val_csv = str(test_csv).replace('.csv', f'_{track}.csv')
        train_df.to_csv(track_train_csv, index=False)
        val_df.to_csv(track_val_csv, index=False)
        
    else:
        # Fallback to original splitting method
        logger.info("No existing split found, creating new split...")
        df = pd.read_csv(csv_path)
        logger.info(f"Total samples: {len(df)}")

        # Random, reproducible split
        train_df = df.sample(frac=train_split, random_state=seed)
        val_df = df.drop(train_df.index)

        logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

        # Save the actual split
        track_train_csv = csv_path.replace('.csv', f'_train_{track}.csv')
        track_val_csv = csv_path.replace('.csv', f'_val_{track}.csv')
        train_df.to_csv(track_train_csv, index=False)
        val_df.to_csv(track_val_csv, index=False)
        logger.info(f"Split data saved: {track_train_csv}, {track_val_csv}")

    # Create datasets with SpecAugment
    logger.info("Creating train dataset with SpecAugment...")
    train_dataset = HakkaASRDatasetWithSpecAugment(
        track_train_csv, processor, track, use_spec_augment, dialect
    )
    train_dataset.set_training_mode(True)  # Enable SpecAugment for training

    logger.info("Creating validation dataset...")
    val_dataset = HakkaASRDatasetWithSpecAugment(
        track_val_csv, processor, track, False, dialect  # Disable SpecAugment for validation
    )
    val_dataset.set_training_mode(False)

    return train_dataset, val_dataset


def train_model(args):
    logger.info(f"Starting enhanced training for {args.track} with SpecAugment")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"SpecAugment enabled: {args.use_spec_augment}")
    logger.info(f"Dialect: {args.dialect}")
    
    model, processor = setup_model_and_processor(args)
    
    train_dataset, val_dataset = create_datasets(
        args.csv_path, processor, args.track, args.use_existing_split, args.train_split, args.seed,
        args.use_spec_augment, args.dialect
    )
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    def compute_metrics_wrapper(pred):
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
    
    # Training arguments
    output_dir_suffix = "_specaugment" if args.use_spec_augment else "_baseline"
    output_dir = f"{args.output_dir}/{args.track}{output_dir_suffix}"
    
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
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_first_step=True,
        logging_strategy="steps",
        seed=args.seed,
    )
    
    logger.info("Training arguments configured:")
    logger.info(f"  Output directory: {output_dir}")
    
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
    logger.info("Starting enhanced training with SpecAugment...")
    logger.info(f"SpecAugment parameters: {args.spec_augment_params if args.use_spec_augment else 'N/A'}")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    final_model_path = f"{output_dir}/final_model"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    logger.info(f"Enhanced training completed! Model saved to {final_model_path}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Enhanced Hakka ASR with SpecAugment Integration")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data/hakka_data.csv", 
                       help="Path to CSV file (will look for _train.csv and _test.csv versions)")
    parser.add_argument("--track", type=str, choices=["track1", "track2"], required=True, 
                       help="Track 1: 客語漢字, Track 2: 客語拼音")
    parser.add_argument("--use_existing_split", action="store_true", default=True,
                       help="Use existing _train.csv and _test.csv files from data_prepare.py")
    parser.add_argument("--train_split", type=float, default=0.8, 
                       help="Training split ratio (only used if --use_existing_split is False)")

    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="formospeech/whisper-large-v3-taiwanese-hakka", 
                       help="Whisper model name")
    parser.add_argument("--language", type=str, default="Chinese", help="Target language")
    parser.add_argument("--task", type=str, default="transcribe", 
                       choices=["transcribe", "translate"], help="Task type")
    parser.add_argument("--output_dir", type=str, default="./hakka_asr_models_enhanced", 
                       help="Output directory")

    # SpecAugment arguments
    parser.add_argument("--use_spec_augment", action="store_true", default=True,
                       help="Enable SpecAugment (default: True)")
    parser.add_argument("--no_spec_augment", action="store_true",
                       help="Disable SpecAugment")
    parser.add_argument("--dialect", type=str, choices=["general", "dapu", "zhaoan"], 
                       default="general", help="Hakka dialect for parameter tuning")
    parser.add_argument("--spec_augment_params", type=str, default="{}",
                       help="JSON string with SpecAugment parameters")

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

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="Resume from checkpoint")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard")
    
    args = parser.parse_args()
    
    # Handle SpecAugment enable/disable
    if args.no_spec_augment:
        args.use_spec_augment = False
    
    # Parse SpecAugment parameters
    try:
        args.spec_augment_params = json.loads(args.spec_augment_params)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON for spec_augment_params, using defaults")
        args.spec_augment_params = {}
    
    set_random_seed(args.seed)
    
    # Setup logging
    logger.info("=" * 70)
    logger.info("ENHANCED HAKKA ASR WITH SPECAUGMENT INTEGRATION")
    logger.info("=" * 70)
    
    # Download NLTK data
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
        logger.info("SpecAugment integration successful!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()