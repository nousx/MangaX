import numpy as np

from ..config import InpainterConfig
from .common import CommonInpainter


class NoneInpainter(CommonInpainter):

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        import cv2
        img_inpainted = np.copy(image)
        
        # Make sure the mask has a single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Binarise the mask; everything > 0 counts
        mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Paint the masked area pure white
        img_inpainted[mask_binary > 0] = np.array([255, 255, 255], np.uint8)
        
        return img_inpainted
