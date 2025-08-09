import argparse
import logging
import pandas as pd
import torch
import torchaudio
import numpy as np
import json
import time
from pathlib import Path
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
import nltk
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hakka_asr_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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


def load_model_and_processor(model_path, device):
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Using device: {device}")
    
    try:
        # Try to load as fine-tuned model first
        if Path(model_path).exists() and Path(model_path).is_dir():
            logger.info("Loading fine-tuned model...")
            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        else:
            # Load as pre-trained model from HuggingFace Hub
            logger.info("Loading pre-trained model from HuggingFace Hub...")
            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        # Set torch dtype based on device
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = model.to(device, dtype=torch_dtype)
        
        logger.info(f"Model loaded successfully with {model.num_parameters():,} parameters")
        logger.info(f"Model dtype: {model.dtype}")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_audio(audio_path, target_sr=16000):
    try:
        audio_array, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)
        
        # Resample to target sample rate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            audio_array = resampler(audio_array)
        
        return audio_array.squeeze().numpy()
        
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        return None


def transcribe_audio(audio_path, model, processor, device, language=None, task="transcribe"):
    """
    Transcribe a single audio file
    """
    # Load audio
    audio = load_audio(audio_path)
    if audio is None:
        return "", 0.0
    
    try:
        # Process audio
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        
        # Generate transcription
        start_time = time.time()
        
        with torch.no_grad():
            # Set generation config if language is specified
            generation_kwargs = {
                "input_features": input_features,
                "max_length": 448,
                "num_beams": 1,
                "do_sample": False,
            }
            
            if language:
                # Force language and task
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language, 
                    task=task
                )
                generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
            
            generated_ids = model.generate(**generation_kwargs)
        
        inference_time = time.time() - start_time
        
        # Decode result
        transcription = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription, inference_time
        
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {e}")
        return "", 0.0


def batch_inference(args):
    """Run batch inference on CSV dataset"""
    logger.info(f"Starting batch inference")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Model: {args.model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, device)
    
    # Load dataset
    df = pd.read_csv(args.csv_path)
    logger.info(f"Loaded {len(df)} samples for inference")
    
    # Determine target column for evaluation
    if args.track == "track1":
        target_column = "客語漢字"
        metric_name = "CER"
        compute_metric = character_error_rate
    elif args.track == "track2":
        target_column = "客語拼音"
        metric_name = "WER"
        compute_metric = word_error_rate
    
    # Run inference
    results = []
    total_time = 0
    
    logger.info("Starting inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        audio_path = row['audio_path']
        reference = str(row[target_column]) if target_column in row else ""
        
        # Transcribe
        prediction, inference_time = transcribe_audio(
            audio_path, model, processor, device, 
            language=args.language, task=args.task
        )
        
        total_time += inference_time
        
        result = {
            "audio_path": audio_path,
            "reference": reference,
            "prediction": prediction,
            "inference_time": inference_time
        }
        
        results.append(result)
        
        if args.verbose:  # Show first 5 examples (if args.verbose and idx < 5:)
            logger.info(f"Sample {idx + 1}:")
            logger.info(f"  Reference: {reference}")
            logger.info(f"  Prediction: {prediction}")
            logger.info(f"  Time: {inference_time:.3f}s")
    
    # Calculate metrics
    references = [r["reference"] for r in results if r["reference"]]
    predictions = [r["prediction"] for r in results if r["reference"]]
    
    if references and predictions:
        if args.track == "track1":
            # corpus-level CER
            total_edits = sum(
                character_error_rate(r, p) * len(r)
                for r, p in zip(references, predictions)
            )
            total_ref_chars = sum(len(r) for r in references)
            avg_metric = total_edits / total_ref_chars * 100
            metric_name = "CER (%)"
        else:
            # corpus-level WER
            total_edits = sum(
                word_error_rate(r, p) * len(r.split())
                for r, p in zip(references, predictions)
            )
            total_ref_words = sum(len(r.split()) for r in references)
            avg_metric = total_edits / total_ref_words * 100
            metric_name = "WER (%)"

        logger.info(f"Corpus-level {metric_name}: {avg_metric:.2f}")

    
    # Summary statistics
    avg_time = total_time / len(results) if results else 0
    logger.info(f"Inference completed!")
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"Average time per file: {avg_time:.3f}s")
    
    # Save results
    if args.output_file:
        output_data = {
            "model_path": args.model_path,
            "track": args.track,
            "total_files": len(results),
            "total_time": total_time,
            "average_time": avg_time,
            "average_metric": avg_metric,
            "metric_name": metric_name,
            "results": results
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {args.output_file}")
    
    return results


def single_inference(args):
    """Run inference on a single audio file"""
    logger.info(f"Single file inference")
    logger.info(f"Audio: {args.audio_path}")
    logger.info(f"Model: {args.model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, device)
    
    # Transcribe
    prediction, inference_time = transcribe_audio(
        args.audio_path, model, processor, device,
        language=args.language, task=args.task
    )
    
    logger.info(f"Transcription: {prediction}")
    logger.info(f"Inference time: {inference_time:.3f}s")
    
    return prediction


def interactive_inference(args):
    """Interactive inference mode"""
    logger.info("Starting interactive inference mode")
    logger.info("Type 'quit' to exit")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, device)
    
    while True:
        try:
            audio_path = input("\nEnter audio file path (or 'quit' to exit): ").strip()
            
            if audio_path.lower() in ['quit', 'exit', 'q']:
                break
            
            if not Path(audio_path).exists():
                print(f"File not found: {audio_path}")
                continue
            
            print("Transcribing...")
            prediction, inference_time = transcribe_audio(
                audio_path, model, processor, device,
                language=args.language, task=args.task
            )
            
            print(f"Transcription: {prediction}")
            print(f"Time: {inference_time:.3f}s")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hakka ASR Inference Script")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "batch", "interactive"],
                       default="batch", help="Inference mode")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="formospeech/whisper-large-v3-taiwanese-hakka",
                       help="Path to fine-tuned model or HuggingFace model name")
    parser.add_argument("--track", type=str, choices=["track1", "track2"], default="track1",
                       help="Track for evaluation: track1 (漢字+CER) or track2 (拼音+WER)")
    
    # Single file mode
    parser.add_argument("--audio_path", type=str,
                       help="Audio file path (for single mode)")
    
    # Batch mode
    parser.add_argument("--csv_path", type=str,
                       default="data/test.csv", help="CSV file path (for batch mode)")
    parser.add_argument("--output_file", type=str,
                       help="Output JSON file path (for batch mode)")
    
    # Generation parameters
    parser.add_argument("--language", type=str, default="Chinese",
                       help="Force language (e.g., 'chinese', 'english')")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                       help="Task type")
    
    # System arguments
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.info("=" * 60)
    logger.info("HAKKA ASR INFERENCE")
    logger.info("=" * 60)
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt')
    
    # Log configuration
    logger.info("Inference configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)
    
    # Validate arguments based on mode
    if args.mode == "single" and not args.audio_path:
        logger.error("Single mode requires --audio_path")
        return
    
    if args.mode == "batch" and not args.csv_path:
        logger.error("Batch mode requires --csv_path")
        return
    
    # Run inference based on mode
    try:
        if args.mode == "single":
            result = single_inference(args)
            
        elif args.mode == "batch":
            results = batch_inference(args)
            
        elif args.mode == "interactive":
            interactive_inference(args)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
