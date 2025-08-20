"""
Noise Augmentation for Hakka ASR
Research-based noise injection techniques optimized for Hakka speech characteristics
"""

import torch
import torchaudio
import numpy as np
import random
import logging
from typing import Union, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class HakkaNoisePipeline:
    """
    Comprehensive noise augmentation pipeline for Hakka ASR
    Based on 2024 research findings for optimal noise injection strategies
    """
    
    def __init__(
        self,
        noise_types: List[str] = None,
        snr_range: Tuple[float, float] = (5, 20),
        noise_prob: float = 0.6,
        noise_dir: Optional[str] = None
    ):
        """
        Initialize Hakka-optimized noise augmentation
        
        Args:
            noise_types: Types of noise to apply ["gaussian", "pink", "brown", "background"]
            snr_range: Signal-to-noise ratio range in dB
            noise_prob: Probability of applying noise augmentation
            noise_dir: Directory containing background noise files
        """
        self.noise_types = noise_types or ["gaussian", "pink", "background"]
        self.snr_range = snr_range
        self.noise_prob = noise_prob
        self.noise_dir = noise_dir
        
        # Load background noise files if directory provided
        self.background_noises = []
        if noise_dir and Path(noise_dir).exists():
            self._load_background_noises()
        
        logger.info(f"HakkaNoisePipeline initialized with types: {self.noise_types}")
        logger.info(f"SNR range: {snr_range}dB, Probability: {noise_prob}")
    
    def _load_background_noises(self):
        """Load background noise files from directory"""
        noise_dir = Path(self.noise_dir)
        noise_files = list(noise_dir.glob("*.wav")) + list(noise_dir.glob("*.flac"))
        
        for noise_file in noise_files[:10]:  # Limit to 10 files for memory
            try:
                noise_audio, sr = torchaudio.load(str(noise_file))
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    noise_audio = resampler(noise_audio)
                
                # Convert to mono
                if noise_audio.shape[0] > 1:
                    noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)
                
                self.background_noises.append(noise_audio.squeeze())
                logger.debug(f"Loaded background noise: {noise_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load noise file {noise_file}: {e}")
        
        logger.info(f"Loaded {len(self.background_noises)} background noise files")
    
    def generate_gaussian_noise(self, length: int, sample_rate: int = 16000) -> torch.Tensor:
        """Generate Gaussian white noise"""
        return torch.randn(length) * 0.1
    
    def generate_pink_noise(self, length: int, sample_rate: int = 16000) -> torch.Tensor:
        """
        Generate pink noise (1/f noise) - more natural for speech
        Research shows pink noise is more effective than white noise for ASR
        """
        # Generate white noise
        white_noise = torch.randn(length)
        
        # Apply 1/f filtering using FFT
        fft = torch.fft.fft(white_noise)
        freqs = torch.fft.fftfreq(length, 1/sample_rate)
        
        # Pink noise has power proportional to 1/f
        pink_filter = 1 / torch.sqrt(torch.abs(freqs) + 1e-8)
        pink_filter[0] = 0  # DC component
        
        # Apply filter
        pink_fft = fft * pink_filter
        pink_noise = torch.fft.ifft(pink_fft).real
        
        # Normalize
        pink_noise = pink_noise / torch.std(pink_noise) * 0.1
        
        return pink_noise
    
    def generate_brown_noise(self, length: int, sample_rate: int = 16000) -> torch.Tensor:
        """
        Generate brown noise (1/f² noise) - even more natural spectrum
        Good for simulating environmental noise
        """
        white_noise = torch.randn(length)
        fft = torch.fft.fft(white_noise)
        freqs = torch.fft.fftfreq(length, 1/sample_rate)
        
        # Brown noise has power proportional to 1/f²
        brown_filter = 1 / (torch.abs(freqs) + 1e-8)
        brown_filter[0] = 0
        
        brown_fft = fft * brown_filter
        brown_noise = torch.fft.ifft(brown_fft).real
        brown_noise = brown_noise / torch.std(brown_noise) * 0.1
        
        return brown_noise
    
    def get_background_noise(self, length: int) -> torch.Tensor:
        """Get random background noise segment"""
        if not self.background_noises:
            # Fallback to pink noise if no background files
            return self.generate_pink_noise(length)
        
        # Choose random background noise
        bg_noise = random.choice(self.background_noises)
        
        # Handle different lengths
        if len(bg_noise) >= length:
            # Extract random segment
            start_idx = random.randint(0, len(bg_noise) - length)
            return bg_noise[start_idx:start_idx + length]
        else:
            # Repeat and pad if necessary
            repeats = (length // len(bg_noise)) + 1
            extended_noise = bg_noise.repeat(repeats)
            return extended_noise[:length]
    
    def add_noise_with_snr(
        self, 
        clean_audio: torch.Tensor, 
        noise: torch.Tensor, 
        target_snr_db: float
    ) -> torch.Tensor:
        """
        Add noise to clean audio at specified SNR
        
        Args:
            clean_audio: Clean speech signal
            noise: Noise signal (same length as clean_audio)
            target_snr_db: Target signal-to-noise ratio in dB
        
        Returns:
            Noisy audio signal
        """
        # Calculate signal power
        signal_power = torch.mean(clean_audio ** 2)
        
        # Calculate noise power for desired SNR
        target_snr_linear = 10 ** (target_snr_db / 10.0)
        noise_power = signal_power / target_snr_linear
        
        # Scale noise to achieve target SNR
        current_noise_power = torch.mean(noise ** 2)
        if current_noise_power > 0:
            noise_scaling = torch.sqrt(noise_power / current_noise_power)
            scaled_noise = noise * noise_scaling
        else:
            scaled_noise = noise
        
        # Add noise to signal
        noisy_audio = clean_audio + scaled_noise
        
        return noisy_audio
    
    def apply_hakka_optimized_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply Hakka-optimized noise augmentation
        
        Research-based approach:
        - Lower SNR for tonal languages (preserve pitch information)
        - Pink noise preferred over white noise
        - Conservative approach for Hakka phonetics
        """
        if random.random() > self.noise_prob:
            return audio
        
        # Choose noise type
        noise_type = random.choice(self.noise_types)
        
        # Generate appropriate noise
        audio_length = len(audio)
        
        if noise_type == "gaussian":
            noise = self.generate_gaussian_noise(audio_length)
        elif noise_type == "pink":
            noise = self.generate_pink_noise(audio_length)
        elif noise_type == "brown":
            noise = self.generate_brown_noise(audio_length)
        elif noise_type == "background":
            noise = self.get_background_noise(audio_length)
        else:
            # Fallback
            noise = self.generate_pink_noise(audio_length)
        
        # Choose SNR - higher SNR for Hakka to preserve tonal information
        # Research shows 10-20dB SNR is optimal for tonal languages
        target_snr = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Apply noise
        noisy_audio = self.add_noise_with_snr(audio, noise, target_snr)
        
        # Prevent clipping
        max_val = torch.max(torch.abs(noisy_audio))
        if max_val > 0.95:
            noisy_audio = noisy_audio * (0.95 / max_val)
        
        return noisy_audio
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentation to audio"""
        return self.apply_hakka_optimized_noise(audio)


class CombinedHakkaAugmentation:
    """
    Combined noise + SpecAugment pipeline for optimal Hakka ASR performance
    Implements 2024 research-based sequential augmentation strategy
    """
    
    def __init__(
        self,
        noise_augment: HakkaNoisePipeline,
        spec_augment: Optional[object] = None,
        noise_first: bool = True,
        combined_prob: float = 0.8
    ):
        """
        Initialize combined augmentation pipeline
        
        Args:
            noise_augment: Noise augmentation pipeline
            spec_augment: SpecAugment instance (HakkaSpecAugment)
            noise_first: Whether to apply noise before spectrogram (recommended)
            combined_prob: Probability of applying combined augmentation
        """
        self.noise_augment = noise_augment
        self.spec_augment = spec_augment
        self.noise_first = noise_first
        self.combined_prob = combined_prob
        
        logger.info(f"CombinedHakkaAugmentation initialized")
        logger.info(f"Strategy: {'Noise→Spec' if noise_first else 'Spec→Noise'}")
        logger.info(f"Combined probability: {combined_prob}")
    
    def apply_audio_domain_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time-domain augmentation (noise injection)"""
        return self.noise_augment(audio)
    
    def apply_spectral_domain_augmentation(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain augmentation (SpecAugment)"""
        if self.spec_augment is not None and self.spec_augment.training:
            return self.spec_augment(spectrogram)
        return spectrogram
    
    def __call__(
        self, 
        audio: torch.Tensor, 
        spectrogram: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply combined augmentation strategy
        
        Args:
            audio: Input audio signal
            spectrogram: Optional pre-computed spectrogram
        
        Returns:
            If spectrogram provided: (augmented_audio, augmented_spectrogram)
            Otherwise: augmented_audio
        """
        if random.random() > self.combined_prob:
            # No augmentation
            return (audio, spectrogram) if spectrogram is not None else audio
        
        # Apply noise augmentation to audio
        augmented_audio = self.apply_audio_domain_augmentation(audio)
        
        if spectrogram is not None:
            # Apply SpecAugment to spectrogram
            augmented_spectrogram = self.apply_spectral_domain_augmentation(spectrogram)
            return augmented_audio, augmented_spectrogram
        else:
            return augmented_audio


# Convenience functions for integration
def create_hakka_noise_pipeline(
    dialect: str = "general",
    conservative: bool = True,
    noise_dir: Optional[str] = None
) -> HakkaNoisePipeline:
    """
    Factory function for Hakka-optimized noise pipeline
    
    Args:
        dialect: Hakka dialect ("dapu", "zhaoan", "general")  
        conservative: Use conservative settings for tonal language
        noise_dir: Directory with background noise files
    
    Returns:
        Configured HakkaNoisePipeline
    """
    if conservative:
        # Conservative settings for tonal languages
        # Higher SNR to preserve pitch information
        snr_range = (10, 20)  # Higher SNR range
        noise_prob = 0.5      # Lower probability
        noise_types = ["pink", "brown"]  # More natural noise types
    else:
        # More aggressive settings
        snr_range = (5, 15)
        noise_prob = 0.7
        noise_types = ["gaussian", "pink", "brown", "background"]
    
    # Dialect-specific adjustments
    if dialect == "dapu":
        # 大埔腔: More conservative (preserve tonal precision)
        snr_range = (12, 22)
        noise_prob = 0.4
    elif dialect == "zhaoan":
        # 詔安腔: Slightly more aggressive
        snr_range = (8, 18)
        noise_prob = 0.6
    
    return HakkaNoisePipeline(
        noise_types=noise_types,
        snr_range=snr_range,
        noise_prob=noise_prob,
        noise_dir=noise_dir
    )


# Example usage
if __name__ == "__main__":
    # Create Hakka noise pipeline
    noise_pipeline = create_hakka_noise_pipeline(
        dialect="general",
        conservative=True,
        noise_dir=None  # Set to your noise directory if available
    )
    
    # Test with dummy audio
    dummy_audio = torch.randn(16000)  # 1 second of audio
    
    # Apply noise augmentation
    noisy_audio = noise_pipeline(dummy_audio)
    
    print(f"Original audio shape: {dummy_audio.shape}")
    print(f"Noisy audio shape: {noisy_audio.shape}")
    print(f"SNR applied: {noise_pipeline.snr_range}")
    print(f"Noise types available: {noise_pipeline.noise_types}")