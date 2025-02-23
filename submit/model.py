import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Handle SMP dependency
try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install segmentation-models-pytorch")
    import segmentation_models_pytorch as smp

import logging

# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class model:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cpu")
        self.transform = T.Compose([
            T.ToTensor(),  # Converts to [0,1] and CHW format
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
        ])
        self.model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b5',
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None  # Outputs raw logits
        ).to(self.device)

    def load(self, path="./"):
        model_path = os.path.join(path, "model.pth")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return self

    def predict(self, X):
        """Process numpy array of shape (3, H, W) in [0-255] range"""
        self.model.eval()
        self.logger.info("Starting prediction process")
        
        try:
            # 1. Validate input
            self.logger.debug(f"Input shape: {X.shape}, dtype: {X.dtype}")
            self.logger.debug(f"Value range: {X.min()} - {X.max()}")
            
            if X.shape[0] != 3:
                raise ValueError(f"Expected CHW format, got {X.shape}")

            # 2. Convert to PIL Image (matches training pipeline)
            pil_image = Image.fromarray(X.transpose(1, 2, 0).astype(np.uint8)).convert("RGB")
            self.logger.debug(f"PIL image: {pil_image.size}, mode: {pil_image.mode}")

            # 3. Apply transforms (critical for model performance)
            tensor_image = self.transform(pil_image)
            self.logger.debug(f"Normalized tensor - Mean: {tensor_image.mean():.3f}, Std: {tensor_image.std():.3f}")

            # 4. Model inference
            batch_tensor = tensor_image.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = self.model(batch_tensor)
            
            # 5. Competition-compliant post-processing
            self.logger.debug(f"Raw output range: {output.min():.3f} - {output.max():.3f}")
            
            # Channel-wise analysis
            squeezed = output.squeeze(0)
            channel_means = squeezed.mean(dim=(1,2)).detach().numpy()
            self.logger.info(f"Channel activations - 0: {channel_means[0]:.3f}, 1: {channel_means[1]:.3f}, 2: {channel_means[2]:.3f}")
            
            # Final prediction
            pred_mask = squeezed.argmax(dim=0).cpu().numpy().astype(np.uint8)
            self.logger.info(f"Predicted classes: {np.unique(pred_mask)}")
            
            return pred_mask
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.exception("Error details:")
            raise