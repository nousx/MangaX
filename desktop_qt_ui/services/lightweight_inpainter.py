"""
轻量级擦除算法接口
为实时预览优化的图像修复算法实现，适配后端的多种inpainter算法
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .erase_config_service import (
    InpainterType,
    get_erase_config_service,
)


@dataclass
class PreviewConfig:
    """预览配置"""
    max_size: int = 512  # Maximum preview size
    quality: float = 0.8  # Preview quality (0.1-1.0)
    cache_enabled: bool = True  # Whether the cache is on
    timeout: float = 5.0  # Timeout (seconds)

class PreviewResult:
    """预览结果"""
    def __init__(self, image: np.ndarray, algorithm: InpainterType, 
                 process_time: float, cached: bool = False):
        self.image = image
        self.algorithm = algorithm
        self.process_time = process_time
        self.cached = cached
        self.timestamp = time.time()

class LightweightInpainter:
    """轻量级图像修复器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_service = get_erase_config_service()
        
        # Preview cache
        self.preview_cache: Dict[str, PreviewResult] = {}
        self.cache_max_size = 10
        
        # Thread pool for asynchronous processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview")
        
        # Mapping of algorithm implementations
        self.algorithm_handlers = {
            InpainterType.NONE: self._inpaint_none,
            InpainterType.ORIGINAL: self._inpaint_original,
            InpainterType.DEFAULT: self._inpaint_simple_blur,  # Simplified AOT
            InpainterType.LAMA_MPE: self._inpaint_simple_blur,  # Simplified LaMa
            InpainterType.LAMA_LARGE: self._inpaint_simple_blur,  # Simplified (not suitable for real time)
            InpainterType.STABLE_DIFFUSION: self._inpaint_simple_blur,  # Simplified (not suitable for real time)
        }
        
        # self.logger.info("Lightweight erase algorithm interface initialised")
    
    def _generate_cache_key(self, image: np.ndarray, mask: np.ndarray, 
                          algorithm: InpainterType, config: PreviewConfig) -> str:
        """生成缓存键"""
        # The hashes of the image and the mask are the cache key
        img_hash = hash(image.tobytes())
        mask_hash = hash(mask.tobytes())
        return f"{algorithm.value}_{img_hash}_{mask_hash}_{config.max_size}_{config.quality}"
    
    def _resize_for_preview(self, image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
        """调整图像尺寸用于预览"""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image.copy(), 1.0
        
        # Work out the scale
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Make sure the size is even (some algorithms require it)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def _inpaint_none(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """无擦除：填充白色"""
        result = image.copy()
        result[mask > 0] = [255, 255, 255]
        return result
    
    def _inpaint_original(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """保持原图不变"""
        return image.copy()
    
    def _inpaint_simple_blur(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """简单模糊填充算法（适合实时预览）"""
        result = image.copy()
        
        # Set the mask area to white
        result[mask > 0] = [255, 255, 255]
        
        # Blur the mask edge slightly to soften hard edges
        if np.any(mask > 0):
            # Create the edge mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            edge_mask = mask_dilated - mask
            
            # Gaussian blur on the edge area
            if np.any(edge_mask > 0):
                blurred = cv2.GaussianBlur(result, (15, 15), 0)
                result[edge_mask > 0] = blurred[edge_mask > 0]
        
        return result
    
    def _inpaint_advanced_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """高级填充算法（更好的质量但仍然轻量）"""
        result = image.copy()
        
        # Fast repair with OpenCV's inpaint function
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        try:
            # Use the fast repair algorithm
            result = cv2.inpaint(image, mask_binary, 3, cv2.INPAINT_TELEA)
        except Exception as e:
            self.logger.warning(f"Advanced fill failed; falling back to simple blur: {e}")
            result = self._inpaint_simple_blur(image, mask)
        
        return result
    
    async def preview_async(self, image: np.ndarray, mask: np.ndarray, 
                          algorithm: Optional[InpainterType] = None,
                          config: Optional[PreviewConfig] = None) -> PreviewResult:
        """异步预览擦除效果"""
        if config is None:
            config = PreviewConfig()
        
        if algorithm is None:
            algorithm = self.config_service.get_current_config().inpainter
        
        start_time = time.time()
        
        # Check the cache
        if config.cache_enabled:
            cache_key = self._generate_cache_key(image, mask, algorithm, config)
            if cache_key in self.preview_cache:
                cached_result = self.preview_cache[cache_key]
                self.logger.debug(f"Using cached result: {algorithm.value}")
                return PreviewResult(
                    cached_result.image.copy(), 
                    algorithm, 
                    time.time() - start_time,
                    cached=True
                )
        
        # Process asynchronously
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            self.executor, 
            self._process_preview, 
            image, mask, algorithm, config
        )
        
        process_time = time.time() - start_time
        result = PreviewResult(result_image, algorithm, process_time)
        
        # Store in the cache
        if config.cache_enabled:
            self._update_cache(cache_key, result)
        
        self.logger.debug(f"Preview completed: {algorithm.value}, elapsed: {process_time:.3f}s")
        return result
    
    def _process_preview(self, image: np.ndarray, mask: np.ndarray,
                        algorithm: InpainterType, config: PreviewConfig) -> np.ndarray:
        """处理预览（在线程池中执行）"""
        try:
            # Resize
            preview_image, scale = self._resize_for_preview(image, config.max_size)
            preview_mask = cv2.resize(mask, 
                                    (preview_image.shape[1], preview_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            preview_mask = np.where(preview_mask > 0, 255, 0).astype(np.uint8)
            
            # Choose the algorithm
            handler = self.algorithm_handlers.get(algorithm, self._inpaint_simple_blur)
            
            # Choose the best implementation for the algorithm and the settings
            if algorithm in [InpainterType.DEFAULT, InpainterType.LAMA_MPE]:
                if config.quality > 0.7:
                    result = self._inpaint_advanced_fill(preview_image, preview_mask)
                else:
                    result = handler(preview_image, preview_mask)
            else:
                result = handler(preview_image, preview_mask)
            
            # When the size was changed, it has to be restored to the original size
            if scale != 1.0:
                original_size = (image.shape[1], image.shape[0])
                result = cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Preview processing failed: {e}")
            # Fall back to the simplest processing
            return self._inpaint_none(image, mask)
    
    def preview_sync(self, image: np.ndarray, mask: np.ndarray,
                    algorithm: Optional[InpainterType] = None,
                    config: Optional[PreviewConfig] = None) -> PreviewResult:
        """同步预览擦除效果"""
        if config is None:
            config = PreviewConfig()
        
        if algorithm is None:
            algorithm = self.config_service.get_current_config().inpainter
        
        start_time = time.time()
        
        # Check the cache
        if config.cache_enabled:
            cache_key = self._generate_cache_key(image, mask, algorithm, config)
            if cache_key in self.preview_cache:
                cached_result = self.preview_cache[cache_key]
                self.logger.debug(f"Using cached result: {algorithm.value}")
                return PreviewResult(
                    cached_result.image.copy(), 
                    algorithm, 
                    time.time() - start_time,
                    cached=True
                )
        
        # Process synchronously
        result_image = self._process_preview(image, mask, algorithm, config)
        process_time = time.time() - start_time
        result = PreviewResult(result_image, algorithm, process_time)
        
        # Store in the cache
        if config.cache_enabled:
            self._update_cache(cache_key, result)
        
        return result
    
    def _update_cache(self, cache_key: str, result: PreviewResult):
        """更新缓存"""
        # Check the cache size limit
        if len(self.preview_cache) >= self.cache_max_size:
            # Remove the oldest cache entry
            oldest_key = min(self.preview_cache.keys(), 
                           key=lambda k: self.preview_cache[k].timestamp)
            del self.preview_cache[oldest_key]
        
        self.preview_cache[cache_key] = result
    
    def clear_cache(self):
        """清空缓存"""
        self.preview_cache.clear()
        # self.logger.info("Preview cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "cache_size": len(self.preview_cache),
            "max_size": self.cache_max_size,
            "algorithms": list(set(result.algorithm for result in self.preview_cache.values())),
            "total_memory_mb": sum(result.image.nbytes for result in self.preview_cache.values()) / (1024 * 1024)
        }
    
    def is_algorithm_suitable_for_preview(self, algorithm: InpainterType) -> bool:
        """检查算法是否适合实时预览"""
        return self.config_service.is_preview_suitable(algorithm)
    
    def get_recommended_preview_algorithm(self) -> InpainterType:
        """获取推荐的预览算法"""
        return self.config_service.get_recommended_preview_algorithm()
    
    def shutdown(self):
        """关闭服务"""
        self.executor.shutdown(wait=True)
        self.clear_cache()
        # self.logger.info("Lightweight erase algorithm interface closed")

# Global service instance
_lightweight_inpainter: Optional[LightweightInpainter] = None

def get_lightweight_inpainter() -> LightweightInpainter:
    """获取轻量级擦除算法接口实例"""
    global _lightweight_inpainter
    if _lightweight_inpainter is None:
        _lightweight_inpainter = LightweightInpainter()
    return _lightweight_inpainter
