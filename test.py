import argparse
import logging
import pandas as pd
import torch
import torchaudio
import numpy as np
import re
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
        logging.FileHandler('hakka_asr_test.log'),
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

    model_dir = Path(model_path)

    try:
        if model_dir.exists() and model_dir.is_dir():
            # 本地微調後模型
            logger.info("Detected local model directory. Loading fine-tuned model...")
            processor = WhisperProcessor.from_pretrained(model_dir.as_posix())
            model = WhisperForConditionalGeneration.from_pretrained(model_dir.as_posix())
        else:
            # HuggingFace Hub 模型
            logger.info("Detected HuggingFace Hub ID. Loading pre-trained model from HF Hub...")
            repo_id = model_path.split("/")[0] + "/" + model_path.split("/")[1] if model_path.count("/") >= 1 else model_path
            subfolder = "/".join(model_path.split("/")[2:]) if model_path.count("/") >= 2 else None

            if subfolder:
                processor = WhisperProcessor.from_pretrained(repo_id, subfolder=subfolder)
                model = WhisperForConditionalGeneration.from_pretrained(repo_id, subfolder=subfolder)
            else:
                processor = WhisperProcessor.from_pretrained(repo_id)
                model = WhisperForConditionalGeneration.from_pretrained(repo_id)

        # 設定 dtype
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


def normalize_text(s: str, track: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # 把連續空白擠成一個
    s = re.sub(r"\s+", " ", s)
    # 全形標點（常見的是中文逗句號）→ 半形
    s = s.replace("，", ",").replace("。", ".")
    if track == "track1":
        # track1（漢字）通常不需要空白；去掉空白避免 CER 受空白影響
        s = s.replace(" ", "")
    else:
        # track2（拼音）一律小寫，避免大小寫差異
        s = s.lower()
    return s


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
        
        input_features = inputs.input_features.to(device, dtype=model.dtype)
        
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
        
        test_time = time.time() - start_time
        
        # Decode result
        transcription = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return transcription, test_time
        
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {e}")
        return "", 0.0


def batch_test(args):
    """Run batch test on CSV dataset"""
    logger.info(f"Starting batch test")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Model: {args.model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, device)
    
    # Load dataset
    df = pd.read_csv(args.csv_path)
    logger.info(f"Loaded {len(df)} samples for test")
    
    # Determine target column for evaluation
    if args.track == "track1":
        target_column = "客語漢字"
        metric_name = "CER"
        compute_metric = character_error_rate
    elif args.track == "track2":
        target_column = "客語拼音"
        metric_name = "WER"
        compute_metric = word_error_rate
    
    # Run test
    results = []
    total_time = 0
    
    logger.info("Starting test...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files", disable=True):
        audio_path = row['audio_path']
        reference = str(row[target_column]) if target_column in row else ""
        
        # Transcribe
        prediction, test_time = transcribe_audio(
            audio_path, model, processor, device, 
            language=args.language, task=args.task
        )
        
        total_time += test_time
        
        reference = normalize_text(reference, args.track)
        prediction = normalize_text(prediction, args.track)

        result = {
            "audio_path": audio_path,
            "reference": reference,
            "prediction": prediction,
            # "test_time": test_time
        }
        
        results.append(result)
        
        if args.verbose:
            logger.info(f"Sample {idx + 1}:")
            if reference == prediction:
                logger.info(f"  Prediction is the same as Reference: {reference}")
            else:
                logger.info(f"  Reference: {reference}")
                logger.info(f"  Prediction: {prediction}")
            # logger.info(f"  Time: {test_time:.3f}s")
    
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
    logger.info(f"Test completed!")
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Total time: {total_time:.3f}s")
    # logger.info(f"Average time per file: {avg_time:.3f}s")
    
    # Save results
    if args.output_file:
        df_results = pd.DataFrame(results, columns=["audio_path", "reference", "prediction"])
        df_results.to_csv(args.output_file, index=False)
        
        logger.info(f"Results saved to: {args.output_file}")
    
    return results


def single_test(args):
    """Run test on a single audio file"""
    logger.info(f"Single file test")
    logger.info(f"Audio: {args.audio_path}")
    logger.info(f"Model: {args.model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, device)
    
    # Transcribe
    prediction, test_time = transcribe_audio(
        args.audio_path, model, processor, device,
        language=args.language, task=args.task
    )
    
    logger.info(f"Transcription: {prediction}")
    # logger.info(f"Test time: {test_time:.3f}s")
    
    return prediction


def interactive_test(args):
    """Interactive test mode"""
    logger.info("Starting interactive test mode")
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
            prediction, test_time = transcribe_audio(
                audio_path, model, processor, device,
                language=args.language, task=args.task
            )
            
            print(f"Transcription: {prediction}")
            print(f"Time: {test_time:.3f}s")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hakka ASR Test Script")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "batch", "interactive"],
                       default="batch", help="Test mode")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="./hakka_asr_models/track1/final_model",
                       help="Path to your fine-tuned model directory")
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
    logger.info("Test configuration:")
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
    
    # Run test based on mode
    try:
        if args.mode == "single":
            result = single_test(args)
            
        elif args.mode == "batch":
            results = batch_test(args)
            
        elif args.mode == "interactive":
            interactive_test(args)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
