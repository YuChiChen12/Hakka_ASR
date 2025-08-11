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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hakka_asr_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HakkaASRDataset(Dataset):
    
    def __init__(self, csv_path, processor, track="track1"):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.track = track
        
        # Determine target column based on track
        self.text_column = "客語漢字" if track == "track1" else "客語拼音"
        
        logger.info(f"Loaded {len(self.df)} samples for {track}")
        logger.info(f"Target column: {self.text_column}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Return a single sample for the given index
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
        
        # Process audio using processor
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.squeeze(0)  # → (seq_len, feat_dim)

        # Process text labels using tokenizer
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",        # 直接拿到 torch.Tensor
            truncation=True,
            max_length=448,
            padding=False,
        ).input_ids.squeeze(0)         # → shape (seq_len,)

        return {
            "input_features": input_features,  # torch.FloatTensor (seq_len, feat_dim)
            "labels": labels,                  # torch.LongTensor  (seq_len,)
        }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Combines multiple samples into a batch with proper padding handling
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Combine features (multiple samples) into a single batch
        """
        # Handle audio inputs and text labels separately as they need different padding methods
        
        # Handle audio inputs - convert to tensors directly
        input_features = [{"input_features": feature["input_features"].cpu().numpy()} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

        # Handle text labels - need special padding treatment
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)

        # Replace padding tokens with -100 so they are ignored during loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if added in previous tokenization step
        # Whisper will add it automatically during training
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
            # 本地微調後模型
            logger.info("Detected local model directory. Loading fine-tuned model...")
            processor = WhisperProcessor.from_pretrained(model_dir.as_posix(), language=args.language, task=args.task)
            model = WhisperForConditionalGeneration.from_pretrained(model_dir.as_posix())
        else:
            # HuggingFace Hub 模型
            logger.info("Detected HuggingFace Hub ID. Loading pre-trained model from HF Hub...")
            processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
            model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    
    logger.info(f"Model and processor loaded successfully")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, processor


def create_datasets(csv_path, processor, track, train_split=0.8, seed=42):
    """Create train/validation datasets with random split and a fixed seed."""
    logger.info(f"Creating datasets from {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Total samples: {len(df)}")

    # Random, reproducible split
    train_df = df.sample(frac=train_split, random_state=seed)
    val_df = df.drop(train_df.index)

    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Save the actual split (for full reproducibility)
    train_csv = csv_path.replace('.csv', f'_train_{track}.csv')
    val_csv   = csv_path.replace('.csv', f'_val_{track}.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    logger.info(f"Split data saved: {train_csv}, {val_csv}")

    logger.info("Creating train dataset...")
    train_dataset = HakkaASRDataset(train_csv, processor, track)

    logger.info("Creating validation dataset...")
    val_dataset = HakkaASRDataset(val_csv, processor, track)

    return train_dataset, val_dataset


def train_model(args):
    logger.info(f"Starting training for {args.track}")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    model, processor = setup_model_and_processor(args)
    
    train_dataset, val_dataset = create_datasets(
        args.csv_path, processor, args.track, args.train_split, args.seed
    )
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    def compute_metrics_wrapper(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
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
    
    # Training arguments using Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/{args.track}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        # gradient_checkpointing=True,
        fp16=args.fp16,
        eval_strategy="steps",
        # eval_on_start=True,
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
        # Seq2Seq specific parameters
        predict_with_generate=True,
        generation_max_length=225,
        logging_first_step=True,
        logging_strategy="steps",
        seed=args.seed,
    )
    
    logger.info("Training arguments configured:")
    for key, value in training_args.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer using Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        # tokenizer=processor.feature_extractor,  # Used for model saving
        processing_class=processor,
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Training will run for {args.max_steps} steps")
    logger.info(f"Evaluation every {args.eval_steps} steps")
    logger.info(f"Model saving every {args.save_steps} steps")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    final_model_path = f"{args.output_dir}/{args.track}/final_model"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    logger.info(f"Training completed! Model saved to {final_model_path}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Final Corrected Whisper Fine-tuning for Hakka ASR")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="data/hakka_data.csv", help="Path to CSV file with audio_path, 客語漢字, 客語拼音 columns")
    parser.add_argument("--track", type=str, choices=["track1", "track2"], required=True, help="Track 1: 客語漢字 (CER), Track 2: 客語拼音 (WER)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio (default: 0.8)")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="formospeech/whisper-large-v3-taiwanese-hakka", help="Whisper model name (default: openai/whisper-small)")
    parser.add_argument("--language", type=str, default="Chinese", help="Target language for fine-tuning (default: chinese)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task type: transcribe or translate (default: transcribe)")
    parser.add_argument("--output_dir", type=str, default="./hakka_asr_models", help="Output directory for models (default: ./hakka_asr_models)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps (default: 500)")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps (default: 5000)")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps (default: 1000)")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps (default: 1000)")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging steps (default: 25)")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers (default: 4)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard logging")
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    
    # Setup logging
    logger.info("=" * 60)
    logger.info("FINAL CORRECTED HAKKA ASR WHISPER FINE-TUNING")
    logger.info("Following HuggingFace Official Method")
    logger.info("=" * 60)
    
    # Download NLTK data for edit distance calculation
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
    logger.info("=" * 60)
    
    # Run training
    try:
        trainer = train_model(args)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
