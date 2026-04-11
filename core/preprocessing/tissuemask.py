
import logging
import numpy as np
import torch
import shutil
import cv2
import os
from skimage import morphology, measure
from scipy import ndimage

from vision_agent.tools import florence2_sam2_instance_segmentation
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.models.architecture.unet import UNetModel
from pillow_heif import register_heif_opener
import core.preprocessing.stainnorm as stainnorm
register_heif_opener()

logger = logging.getLogger(__name__)


class FlorenceTissueMaskExtractor:
    def __init__(self, unet_model_path: str = "", unet_device: str = "cuda"):
        # Define default and fallback prompts
        self.default_prompt = "tissue,stain"
        self.backup_prompts = ["tissue,stain", "tissue", "cell,tissue", "histology"]
        self.unet_model_path = unet_model_path
        self.unet_device = unet_device

    def extract(self, image: np.ndarray, artefacts: bool) -> np.ndarray:
        """
        Extracts the tissue mask from an image using instance segmentation or fallback methods.

        Extraction order:
          1. Florence-2 + SAM2 prompt-based instance segmentation.
          2. If that fails and a UNet model path is provided, the UNet extractor.
          3. Final fallback: Otsu threshold with morphological cleanup.

        Args:
            image (np.ndarray): Input RGB image.
            artefacts (bool): When True, return only the first (largest) segment mask
                so that control tissue artefacts are isolated.

        Returns:
            np.ndarray: Binary tissue mask (uint8, values 0 or 255).
        """
        # Try instance segmentation first
        segments = self._segment_with_prompts(image, self.default_prompt)

        if not segments:
            for prompt in self.backup_prompts:
                segments = self._segment_with_prompts(image, prompt)
                if segments:
                    break
                else:
                    stain = stainnorm.StainNormalizer()
                    norm, h, e = stain.process(image)
                    segments = self._segment_with_prompts(norm, prompt)

        if artefacts:
            if segments:
                return (segments[0]['mask'] * 255).astype(np.uint8)
            # No segments found for artefact extraction — fall through to UNet / fallback
        else:
            if segments:
                combined_mask = np.zeros_like(segments[0]['mask'], dtype=np.uint8)
                for segment in segments:
                    combined_mask = np.maximum(combined_mask, (segment['mask'] * 255).astype(np.uint8))
                return combined_mask

        # Try UNet-based extraction if a model path has been provided
        if self.unet_model_path:
            return self._unet_mask(image)

        # Final fallback
        return self._fallback_mask(image)

    @staticmethod
    def _segment_with_prompts(image: np.ndarray, prompt: str):
        try:
            return florence2_sam2_instance_segmentation(prompt, image)
        except Exception:
            return []

    def _unet_mask(self, image: np.ndarray) -> np.ndarray:
        """Extract tissue mask using the UNet model, falling back to Otsu on error."""
        try:
            extractor = UNetTissueMaskExtractor(
                model_path=self.unet_model_path,
                device=self.unet_device,
            )
            mask = extractor.extract_masks(image)
            if mask is not None:
                return mask
        except Exception as exc:
            logger.warning("UNet tissue mask extraction failed: %s", exc)
        return self._fallback_mask(image)

    def _fallback_mask(self, image: np.ndarray) -> np.ndarray:
        """Fallback method using Otsu threshold and morphology."""
        logger.info("Applying fallback tissue mask extraction.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, threshold_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_binary = (mask > 0).astype(np.uint8)

        # Invert to match tissue as foreground
        mask_binary = 1 - mask_binary

        # Extract largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels <= 1:
            return mask_binary

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype(np.uint8)

        return largest_component_mask
class UNetTissueMaskExtractor:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path (str): Path to the pretrained UNet checkpoint.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()

    @staticmethod
    def convert_pytorch_checkpoint(net_state_dict):
        """Convert checkpoint from DataParallel to single-GPU format."""
        variable_name_list = list(net_state_dict.keys())
        is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
        if is_in_parallel_mode:
            net_state_dict = {
                ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
            }
        return net_state_dict

    @staticmethod
    def post_processing_mask(mask: np.ndarray) -> np.ndarray:
        """Fill holes and keep only the largest object in the binary mask."""
        mask = ndimage.binary_fill_holes(mask, structure=np.ones((3, 3))).astype(int)
        label_img = measure.label(mask)

        if len(np.unique(label_img)) > 2:
            regions = measure.regionprops(label_img)
            mask = mask.astype(bool)
            all_area = [r.area for r in regions]
            second_max = max([a for a in all_area if a != max(all_area)], default=0)
            mask = morphology.remove_small_objects(mask, min_size=second_max + 1)

        return mask.astype(np.uint8)

    def _load_model(self):
        """Load and return the UNet model."""
        if self.device == "cuda":
            pretrained = torch.load(self.model_path, map_location='cuda')
        else:
            pretrained = torch.load(self.model_path, map_location='cpu')

        pretrained = self.convert_pytorch_checkpoint(pretrained)
        model = UNetModel(num_input_channels=3, num_output_channels=3)
        model.load_state_dict(pretrained)
        return model

    def extract_masks(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a tissue mask for a single image using UNet segmentation.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            np.ndarray: Processed binary tissue mask.
        """
        global_save_dir = "./tmp/"
        save_dir = os.path.join(global_save_dir, 'tissue_mask')

        # Clean up and create fresh directories
        if os.path.exists(global_save_dir):
            shutil.rmtree(global_save_dir)
        os.makedirs(save_dir)

        # Prepare RGB input from grayscale
        image_rgb = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
        # Save images
        image_path = os.path.join(global_save_dir, 'image.png')
        cv2.imwrite(image_path, image_rgb)

        # Create segmentor and predict
        segmentor = SemanticSegmentor(
            model=self.model,
            pretrained_model="unet_tissue_mask_tsef",
            num_loader_workers=4,
            batch_size=4,
        )

        output = segmentor.predict(
            [image_path],
            save_dir=save_dir,
            mode="tile",
            resolution=1.0,
            units="baseline",
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            device=self.device,
            crash_on_exception=True,
        )

        # Load and process masks
        mask = np.load(output[0][1] + ".raw.0.npy")
        mask = np.argmax(mask, axis=-1) == 2
        mask = self.post_processing_mask(mask)

        return mask
