"""
Configuration management service for the erase algorithms.
Reads the inpainter settings from the configuration file and manages the choice between several erase algorithms
"""
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class InpainterType(Enum):
    """Erase algorithm type enumeration"""
    DEFAULT = "default"  # AOT algorithm
    LAMA_LARGE = "lama_large"  # Lama Large algorithm
    LAMA_MPE = "lama_mpe"  # Lama MPE algorithm
    STABLE_DIFFUSION = "sd"  # Stable Diffusion algorithm
    NONE = "none"  # No erasing; fill with white
    ORIGINAL = "original"  # Keep the original image unchanged

class InpaintPrecision(Enum):
    """Inpainting precision enumeration"""
    FP32 = "fp32"
    FP16 = "fp16" 
    BF16 = "bf16"

@dataclass
class InpainterConfig:
    """Erase algorithm configuration"""
    inpainter: InpainterType = InpainterType.LAMA_LARGE
    inpainting_size: int = 2048
    inpainting_precision: InpaintPrecision = InpaintPrecision.BF16
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary form"""
        return {
            "inpainter": self.inpainter.value,
            "inpainting_size": self.inpainting_size,
            "inpainting_precision": self.inpainting_precision.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InpainterConfig':
        """Create a configuration object from a dictionary"""
        return cls(
            inpainter=InpainterType(data.get("inpainter", "lama_large")),
            inpainting_size=data.get("inpainting_size", 2048),
            inpainting_precision=InpaintPrecision(data.get("inpainting_precision", "bf16"))
        )

@dataclass  
class AlgorithmInfo:
    """Algorithm information"""
    name: str
    display_name: str
    description: str
    supports_gpu: bool = True
    supports_precision: bool = True
    preview_suitable: bool = True  # Whether it is suitable for a live preview

class EraseConfigService:
    """Configuration management service for the erase algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Current configuration
        self.current_config = InpainterConfig()
        
        # Mapping of algorithm information
        self.algorithm_info = {
            InpainterType.DEFAULT: AlgorithmInfo(
                name="default",
                display_name="Default (AOT)",
                description="The default AOT inpainting algorithm, fairly fast",
                supports_gpu=True,
                supports_precision=False,
                preview_suitable=True
            ),
            InpainterType.LAMA_LARGE: AlgorithmInfo(
                name="lama_large", 
                display_name="Lama Large",
                description="High-quality Lama inpainting algorithm, best results",
                supports_gpu=True,
                supports_precision=True,
                preview_suitable=False  # The model is large and not suitable for a live preview
            ),
            InpainterType.LAMA_MPE: AlgorithmInfo(
                name="lama_mpe",
                display_name="Lama MPE", 
                description="Lightweight Lama algorithm, a balance of speed and quality",
                supports_gpu=True,
                supports_precision=True,
                preview_suitable=True
            ),
            InpainterType.STABLE_DIFFUSION: AlgorithmInfo(
                name="sd",
                display_name="Stable Diffusion",
                description="Inpainting algorithm based on a diffusion model, very high quality but slow",
                supports_gpu=True,
                supports_precision=True,
                preview_suitable=False  # Too slow for a live preview
            ),
            InpainterType.NONE: AlgorithmInfo(
                name="none",
                display_name="No erasing",
                description="Nothing is erased; the mask area is filled with white",
                supports_gpu=False,
                supports_precision=False,
                preview_suitable=True
            ),
            InpainterType.ORIGINAL: AlgorithmInfo(
                name="original", 
                display_name="Keep original",
                description="The original image is kept unchanged, without any processing",
                supports_gpu=False,
                supports_precision=False,
                preview_suitable=True
            )
        }
        
    
    def load_config_from_file(self, config_path: str) -> bool:
        """Load the settings from the configuration file"""
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Configuration file does not exist: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract the inpainter configuration
            inpainter_data = config_data.get("inpainter", {})
            if inpainter_data:
                self.current_config = InpainterConfig.from_dict(inpainter_data)
                self.logger.info(f"Loaded erase algorithm settings from configuration: {self.current_config.inpainter.value}")
                return True
            else:
                self.logger.warning("No inpainter settings found in configuration file")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            return False
    
    def save_config_to_file(self, config_path: str) -> bool:
        """Save the configuration to the file"""
        try:
            # Read the existing configuration
            config_data = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # Update the inpainter configuration
            config_data["inpainter"] = self.current_config.to_dict()
            
            # Save the configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration file: {e}")
            return False
    
    def get_algorithm_list(self) -> List[Dict[str, Any]]:
        """Get the list of available algorithms"""
        return [
            {
                "type": algo_type,
                "name": info.name,
                "display_name": info.display_name,
                "description": info.description,
                "supports_gpu": info.supports_gpu,
                "supports_precision": info.supports_precision,
                "preview_suitable": info.preview_suitable
            }
            for algo_type, info in self.algorithm_info.items()
        ]
    
    def get_preview_suitable_algorithms(self) -> List[InpainterType]:
        """Get the algorithms suited to live preview"""
        return [
            algo_type for algo_type, info in self.algorithm_info.items()
            if info.preview_suitable
        ]
    
    def set_algorithm(self, algorithm: InpainterType):
        """Set the current algorithm"""
        self.current_config.inpainter = algorithm
        self.logger.info(f"Switching erase algorithm: {algorithm.value}")
    
    def set_inpainting_size(self, size: int):
        """Set the inpainting size"""
        if size < 512 or size > 4096:
            raise ValueError("The inpainting size must be between 512 and 4096")
        self.current_config.inpainting_size = size
        self.logger.info(f"Setting inpainting size: {size}")
    
    def set_precision(self, precision: InpaintPrecision):
        """Set the inpainting precision"""
        if not self.algorithm_info[self.current_config.inpainter].supports_precision:
            self.logger.warning(f"Current algorithm does not support precision settings: {self.current_config.inpainter.value}")
            return
        self.current_config.inpainting_precision = precision
        self.logger.info(f"Setting inpainting precision: {precision.value}")
    
    def get_current_config(self) -> InpainterConfig:
        """Get the current configuration"""
        return self.current_config
    
    def get_algorithm_info(self, algorithm: InpainterType) -> Optional[AlgorithmInfo]:
        """Get the information of an algorithm"""
        return self.algorithm_info.get(algorithm)
    
    def is_preview_suitable(self, algorithm: InpainterType) -> bool:
        """Check whether an algorithm is suited to live preview"""
        info = self.algorithm_info.get(algorithm)
        return info.preview_suitable if info else False
    
    def get_recommended_preview_algorithm(self) -> InpainterType:
        """Get the recommended preview algorithm"""
        # Recommend first the algorithms that suit a preview and give good results
        preview_algorithms = self.get_preview_suitable_algorithms()
        
        # Order of priority: lama_mpe > default > none > original
        priority_order = [
            InpainterType.LAMA_MPE,
            InpainterType.DEFAULT, 
            InpainterType.NONE,
            InpainterType.ORIGINAL
        ]
        
        for algo in priority_order:
            if algo in preview_algorithms:
                return algo
        
        # When none is available, return the first one
        return preview_algorithms[0] if preview_algorithms else InpainterType.NONE

# Global service instance
_erase_config_service: Optional[EraseConfigService] = None

def get_erase_config_service() -> EraseConfigService:
    """Get the instance of the erase configuration service"""
    global _erase_config_service
    if _erase_config_service is None:
        _erase_config_service = EraseConfigService()
    return _erase_config_service