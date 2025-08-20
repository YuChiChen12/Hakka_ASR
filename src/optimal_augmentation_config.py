"""
Optimal Augmentation Configuration for Hakka ASR
Based on 2024 research findings and tonal language considerations
"""

from typing import Dict, Any, Tuple, List
import random
import logging

logger = logging.getLogger(__name__)


class OptimalAugmentationConfig:
    """
    Research-based optimal augmentation configurations for Hakka ASR
    Implements findings from 2024 literature on tonal language ASR augmentation
    """
    
    # Standard SpecAugment parameters based on research
    SPECAUGMENT_STANDARD = {
        "time_mask_param": 80,          # Reduced from 100 for tonal languages
        "freq_mask_param": 27,          # Standard MEL spectrogram masking
        "time_mask_num": 1,             # Conservative single masking
        "freq_mask_num": 1,             # Conservative single masking
        "time_warp_param": 0,           # Disabled for stability
        "mask_value": 0.0,              # Zero masking value
    }
    
    # Dialect-specific optimizations
    DIALECT_CONFIGS = {
        "dapu": {
            "time_mask_param": 64,      # 80 * 0.8 - more conservative
            "freq_mask_param": 27,
            "time_mask_num": 1,
            "freq_mask_num": 1,
            "time_warp_param": 0,
            "description": "Conservative masking for tonal precision (大埔腔)"
        },
        "zhaoan": {
            "time_mask_param": 88,      # 80 * 1.1 - slightly more aggressive
            "freq_mask_param": 27,
            "time_mask_num": 1,
            "freq_mask_num": 1,
            "time_warp_param": 0,
            "description": "Balanced approach for phonetic diversity (詔安腔)"
        },
        "general": {
            "time_mask_param": 80,      # Standard configuration
            "freq_mask_param": 27,
            "time_mask_num": 1,
            "freq_mask_num": 1,
            "time_warp_param": 0,
            "description": "General Hakka configuration"
        }
    }
    
    # Noise augmentation configurations for tonal languages
    NOISE_CONFIGS = {
        "conservative": {
            "noise_types": ["pink", "brown"],
            "snr_range": (10, 20),      # Higher SNR to preserve tonal info
            "noise_prob": 0.5,          # Conservative probability
            "description": "Conservative for tonal language preservation"
        },
        "dapu": {
            "noise_types": ["pink", "brown"],
            "snr_range": (12, 22),      # Even higher SNR for Dapu
            "noise_prob": 0.4,          # More conservative
            "description": "Optimized for Dapu dialect characteristics"
        },
        "zhaoan": {
            "noise_types": ["pink", "brown", "background"],
            "snr_range": (8, 18),       # Slightly lower SNR
            "noise_prob": 0.6,          # More aggressive
            "description": "Optimized for Zhao'an dialect characteristics"
        },
        "aggressive": {
            "noise_types": ["gaussian", "pink", "brown", "background"],
            "snr_range": (5, 15),       # Standard range
            "noise_prob": 0.7,          # Higher probability
            "description": "More aggressive augmentation for robust training"
        }
    }
    
    # Progressive training schedule
    PROGRESSIVE_SCHEDULE = {
        "early": {  # Epochs 1-5
            "spec_aug_prob": 0.5,
            "noise_prob": 0.3,
            "time_mask_scale": 0.7,
            "freq_mask_scale": 0.7,
            "description": "Gentle augmentation for initial training"
        },
        "mid": {    # Epochs 6-15
            "spec_aug_prob": 0.8,
            "noise_prob": 0.6,
            "time_mask_scale": 1.0,
            "freq_mask_scale": 1.0,
            "description": "Standard augmentation strength"
        },
        "late": {   # Epochs 16+
            "spec_aug_prob": 0.9,
            "noise_prob": 0.7,
            "time_mask_scale": 1.2,
            "freq_mask_scale": 1.2,
            "description": "Enhanced augmentation for fine-tuning"
        }
    }
    
    # Dataset balancing ratios based on analysis
    # Format: (SpecAug prob, Noise prob, Sampling multiplier)
    BALANCING_RATIOS = {
        "dapu_male": (0.9, 0.7, 1.4),      # Most augmentation needed
        "dapu_female": (0.8, 0.6, 1.1),    # Moderate augmentation
        "zhaoan_male": (0.8, 0.6, 1.1),    # Moderate augmentation
        "zhaoan_female": (0.7, 0.5, 0.9),  # Least augmentation needed
    }
    
    # Quality control thresholds
    QUALITY_CONTROL = {
        "max_mask_ratio": 0.25,            # Maximum masking percentage
        "min_snr": 5,                       # Minimum SNR threshold
        "max_snr": 25,                      # Maximum SNR threshold
        "augmentation_coverage": 0.8,       # Target coverage ratio
        "dialect_balance_tolerance": 0.05,  # Acceptable dialect imbalance
        "gender_balance_tolerance": 0.1,    # Acceptable gender imbalance
    }

    def __init__(self):
        """Initialize the configuration manager"""
        logger.info("OptimalAugmentationConfig initialized")
    
    def get_specaugment_config(self, dialect: str = "general", 
                             track: str = "track1") -> Dict[str, Any]:
        """
        Get SpecAugment configuration for specific dialect and track
        
        Args:
            dialect: "dapu", "zhaoan", or "general"
            track: "track1" (漢字) or "track2" (拼音)
        
        Returns:
            Dictionary with SpecAugment parameters
        """
        if dialect not in self.DIALECT_CONFIGS:
            logger.warning(f"Unknown dialect {dialect}, using general config")
            dialect = "general"
        
        config = self.DIALECT_CONFIGS[dialect].copy()
        
        # Track-specific adjustments
        if track == "track2":  # 拼音 track needs more careful augmentation
            config["time_mask_param"] = int(config["time_mask_param"] * 0.9)
            logger.info(f"Applied track2 adjustment for {dialect}")
        
        config["dialect"] = dialect
        config["track"] = track
        
        logger.info(f"SpecAugment config for {dialect}-{track}: {config}")
        return config
    
    def get_noise_config(self, dialect: str = "general", 
                        conservative: bool = True) -> Dict[str, Any]:
        """
        Get noise augmentation configuration
        
        Args:
            dialect: "dapu", "zhaoan", or "general"
            conservative: Whether to use conservative settings
        
        Returns:
            Dictionary with noise augmentation parameters
        """
        if dialect == "dapu":
            config_key = "dapu"
        elif dialect == "zhaoan":
            config_key = "zhaoan"
        elif conservative:
            config_key = "conservative"
        else:
            config_key = "aggressive"
        
        config = self.NOISE_CONFIGS[config_key].copy()
        config["dialect"] = dialect
        
        logger.info(f"Noise config for {dialect}: {config}")
        return config
    
    def get_progressive_config(self, epoch: int) -> Dict[str, Any]:
        """
        Get progressive training configuration based on epoch
        
        Args:
            epoch: Current training epoch
        
        Returns:
            Dictionary with progressive training parameters
        """
        if epoch <= 5:
            stage = "early"
        elif epoch <= 15:
            stage = "mid"
        else:
            stage = "late"
        
        config = self.PROGRESSIVE_SCHEDULE[stage].copy()
        config["stage"] = stage
        config["epoch"] = epoch
        
        logger.info(f"Progressive config for epoch {epoch} ({stage}): {config}")
        return config
    
    def get_balancing_config(self, dialect: str, gender: str) -> Tuple[float, float, float]:
        """
        Get data balancing configuration
        
        Args:
            dialect: "dapu" or "zhaoan"
            gender: "male" or "female"
        
        Returns:
            Tuple of (SpecAug probability, Noise probability, Sampling multiplier)
        """
        key = f"{dialect}_{gender}"
        if key not in self.BALANCING_RATIOS:
            logger.warning(f"Unknown balancing key {key}, using default")
            return (0.8, 0.6, 1.0)
        
        config = self.BALANCING_RATIOS[key]
        logger.info(f"Balancing config for {key}: SpecAug={config[0]}, Noise={config[1]}, Sampling={config[2]}")
        return config
    
    def validate_quality(self, mask_ratio: float, snr: float) -> bool:
        """
        Validate augmentation quality
        
        Args:
            mask_ratio: Proportion of masked spectrogram
            snr: Signal-to-noise ratio
        
        Returns:
            Boolean indicating if quality is acceptable
        """
        qc = self.QUALITY_CONTROL
        
        if mask_ratio > qc["max_mask_ratio"]:
            logger.warning(f"Mask ratio {mask_ratio:.2f} exceeds threshold {qc['max_mask_ratio']}")
            return False
        
        if not (qc["min_snr"] <= snr <= qc["max_snr"]):
            logger.warning(f"SNR {snr:.1f}dB outside acceptable range [{qc['min_snr']}, {qc['max_snr']}]")
            return False
        
        return True
    
    def get_comprehensive_config(self, dialect: str = "general", track: str = "track1",
                               epoch: int = 1, gender: str = "female",
                               conservative: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive configuration combining all aspects
        
        Args:
            dialect: Target dialect
            track: Training track
            epoch: Current epoch
            gender: Speaker gender
            conservative: Conservative settings flag
        
        Returns:
            Complete configuration dictionary
        """
        config = {
            "specaugment": self.get_specaugment_config(dialect, track),
            "noise": self.get_noise_config(dialect, conservative),
            "progressive": self.get_progressive_config(epoch),
            "balancing": dict(zip(
                ["spec_aug_prob", "noise_prob", "sampling_multiplier"],
                self.get_balancing_config(dialect, gender)
            )),
            "quality_control": self.QUALITY_CONTROL.copy(),
            "meta": {
                "dialect": dialect,
                "track": track,
                "epoch": epoch,
                "gender": gender,
                "conservative": conservative,
                "config_version": "2024.1.0"
            }
        }
        
        logger.info(f"Generated comprehensive config for {dialect}-{gender}-{track} epoch {epoch}")
        return config


# Factory functions for easy usage
def create_optimal_specaugment_config(dialect: str, track: str) -> Dict[str, Any]:
    """Factory function for SpecAugment configuration"""
    manager = OptimalAugmentationConfig()
    return manager.get_specaugment_config(dialect, track)


def create_optimal_noise_config(dialect: str, conservative: bool = True) -> Dict[str, Any]:
    """Factory function for noise augmentation configuration"""
    manager = OptimalAugmentationConfig()
    return manager.get_noise_config(dialect, conservative)


def get_research_based_ratios() -> Dict[str, Tuple[float, float, float]]:
    """Get research-based augmentation ratios for dataset balancing"""
    return OptimalAugmentationConfig.BALANCING_RATIOS.copy()


# Configuration validation
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Boolean indicating validity
    """
    required_keys = ["time_mask_param", "freq_mask_param", "time_mask_num", "freq_mask_num"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
        
        if not isinstance(config[key], (int, float)) or config[key] < 0:
            logger.error(f"Invalid value for {key}: {config[key]}")
            return False
    
    logger.info("Configuration validation passed")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration manager
    config_manager = OptimalAugmentationConfig()
    
    # Test different configurations
    print("=== SpecAugment Configurations ===")
    for dialect in ["dapu", "zhaoan", "general"]:
        for track in ["track1", "track2"]:
            config = config_manager.get_specaugment_config(dialect, track)
            print(f"{dialect}-{track}: time_mask={config['time_mask_param']}, freq_mask={config['freq_mask_param']}")
    
    print("\n=== Noise Configurations ===")
    for dialect in ["dapu", "zhaoan", "general"]:
        config = config_manager.get_noise_config(dialect, conservative=True)
        print(f"{dialect}: SNR={config['snr_range']}, types={config['noise_types']}, prob={config['noise_prob']}")
    
    print("\n=== Progressive Training ===")
    for epoch in [1, 8, 20]:
        config = config_manager.get_progressive_config(epoch)
        print(f"Epoch {epoch}: stage={config['stage']}, spec_aug_prob={config['spec_aug_prob']}")
    
    print("\n=== Balancing Ratios ===")
    for dialect in ["dapu", "zhaoan"]:
        for gender in ["male", "female"]:
            ratios = config_manager.get_balancing_config(dialect, gender)
            print(f"{dialect}_{gender}: SpecAug={ratios[0]}, Noise={ratios[1]}, Sampling={ratios[2]}")
    
    print("\n=== Comprehensive Configuration Example ===")
    comprehensive = config_manager.get_comprehensive_config(
        dialect="dapu", track="track1", epoch=10, gender="male", conservative=True
    )
    print(f"Comprehensive config keys: {list(comprehensive.keys())}")
    print(f"Metadata: {comprehensive['meta']}")