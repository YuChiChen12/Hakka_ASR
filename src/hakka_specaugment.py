"""
SpecAugment Integration for Hakka ASR Training
Optimized for Whisper fine-tuning pipeline with Hakka language characteristics
"""

import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging


class HakkaSpecAugment(nn.Module):
    """
    SpecAugment specifically tuned for Hakka language characteristics
    Integrates seamlessly with Whisper feature extraction pipeline
    """
    
    def __init__(
        self,
        time_mask_param: int = 80,      # Reduced for tonal language
        freq_mask_param: int = 27,      # Standard for mel spectrograms
        time_mask_num: int = 1,
        freq_mask_num: int = 1,
        time_warp_param: int = 0,       # DISABLED - set to 0 to disable time warping
        mask_value: float = 0.0,
        dialect: str = "general",       # "dapu", "zhaoan", "general"
        track: str = "track1"           # "track1" (漢字), "track2" (拼音)
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.time_warp_param = time_warp_param
        self.mask_value = mask_value
        self.dialect = dialect
        self.track = track
        
        # Dialect-specific adjustments
        self._adjust_for_dialect()
        
        logging.info(f"HakkaSpecAugment initialized for {dialect} dialect, {track}")
        logging.info(f"Parameters: time_mask={time_mask_param}, freq_mask={freq_mask_param}, warp={time_warp_param}")
    
    def _adjust_for_dialect(self):
        """Adjust parameters based on Hakka dialect characteristics"""
        if self.dialect == "dapu":
            # 大埔腔: More conservative time masking due to tonal precision
            self.time_mask_param = int(self.time_mask_param * 0.8)
        elif self.dialect == "zhaoan":
            # 詔安腔: Slightly more aggressive due to different phonetic structure
            self.time_mask_param = int(self.time_mask_param * 1.1)
        
        # Track-specific adjustments
        if self.track == "track2":  # 拼音 track needs more careful augmentation
            self.time_mask_param = int(self.time_mask_param * 0.9)
    
    def time_masking(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking to spectrogram"""
        spec = spec.clone()
        _, time_len = spec.shape
        
        for _ in range(self.time_mask_num):
            t = random.randint(0, min(self.time_mask_param, int(time_len * 0.15)))
            if t == 0:
                continue
            t0 = random.randint(0, max(0, time_len - t))
            spec[:, t0:t0+t] = self.mask_value
        
        return spec
    
    def frequency_masking(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to spectrogram"""
        spec = spec.clone()
        freq_len, _ = spec.shape
        
        for _ in range(self.freq_mask_num):
            f = random.randint(0, min(self.freq_mask_param, int(freq_len * 0.3)))
            if f == 0:
                continue
            f0 = random.randint(0, max(0, freq_len - f))
            spec[f0:f0+f, :] = self.mask_value
        
        return spec
    
    def time_warping(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time warping - DISABLED for stability"""
        # Time warping is disabled to prevent potential errors
        # Return the original spectrogram unchanged
        return spec
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment transforms
        
        Args:
            spectrogram: Input mel spectrogram [freq_bins, time_frames]
        
        Returns:
            Augmented spectrogram with same shape
        """
        if not self.training:
            return spectrogram
        
        # Apply transforms: freq mask -> time mask (time warp disabled)
        spec = spectrogram.clone()
        spec = self.frequency_masking(spec)
        spec = self.time_masking(spec)
        
        return spec


class AdaptiveHakkaSpecAugment(HakkaSpecAugment):
    """
    Adaptive SpecAugment that adjusts parameters during training
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.base_time_mask = self.time_mask_param
        self.base_freq_mask = self.freq_mask_param
    
    def update_epoch(self, epoch: int):
        """Update augmentation strength based on training epoch"""
        self.epoch = epoch
        
        # Progressive augmentation: start gentle, increase strength
        if epoch < 5:
            # Early training: reduced augmentation
            self.time_mask_param = int(self.base_time_mask * 0.5)
            self.freq_mask_param = int(self.base_freq_mask * 0.5)
        elif epoch < 15:
            # Mid training: standard augmentation
            self.time_mask_param = self.base_time_mask
            self.freq_mask_param = self.base_freq_mask
        else:
            # Late training: stronger augmentation
            self.time_mask_param = int(self.base_time_mask * 1.2)
            self.freq_mask_param = int(self.base_freq_mask * 1.2)


class HakkaAudioTransform:
    """
    Audio preprocessing pipeline with integrated SpecAugment for Hakka ASR
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        use_spec_augment: bool = True,
        spec_augment_params: Optional[Dict[str, Any]] = None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.use_spec_augment = use_spec_augment
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate//2
        )
        
        # SpecAugment
        if use_spec_augment:
            params = spec_augment_params or {}
            self.spec_augment = HakkaSpecAugment(**params)
        else:
            self.spec_augment = None
    
    def __call__(self, waveform: torch.Tensor, apply_augment: bool = True) -> torch.Tensor:
        """
        Process audio waveform to mel spectrogram with optional SpecAugment
        
        Args:
            waveform: Input audio waveform [channels, samples] or [samples]
            apply_augment: Whether to apply SpecAugment (should be False during evaluation)
        
        Returns:
            Processed mel spectrogram [n_mels, time_frames]
        """
        # Ensure proper shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = mel_spec.squeeze(0)  # Remove channel dimension
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Apply SpecAugment during training
        if apply_augment and self.use_spec_augment and self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        
        return mel_spec


def create_hakka_spec_augment(dialect: str, track: str) -> HakkaSpecAugment:
    """
    Factory function to create optimized SpecAugment for specific Hakka configuration
    
    Args:
        dialect: "dapu", "zhaoan", or "general"
        track: "track1" (客語漢字) or "track2" (客語拼音)
    
    Returns:
        Configured HakkaSpecAugment instance
    """
    # Base parameters optimized for Hakka (time warp disabled)
    base_params = {
        "time_mask_param": 80,
        "freq_mask_param": 27,
        "time_mask_num": 1,
        "freq_mask_num": 1,
        "time_warp_param": 0,  # DISABLED for stability
        "dialect": dialect,
        "track": track
    }
    
    return HakkaSpecAugment(**base_params)


# Example usage and integration points
if __name__ == "__main__":
    # Example: Create SpecAugment for Dapu dialect, Track 1
    spec_aug = create_hakka_spec_augment("dapu", "track1")
    
    # Example: Process a dummy spectrogram
    dummy_spec = torch.randn(80, 300)  # [n_mels, time_frames]
    
    # Apply augmentation (training mode)
    spec_aug.train()
    augmented_spec = spec_aug(dummy_spec)
    
    print(f"Original shape: {dummy_spec.shape}")
    print(f"Augmented shape: {augmented_spec.shape}")
    print(f"Augmentation applied: {not torch.equal(dummy_spec, augmented_spec)}")