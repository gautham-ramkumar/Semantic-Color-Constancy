"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2024-06-15
Applying Tint on Images - Data Preparation Part 3
CS7180 - Advanced Perception
"""

import os
import zipfile
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil


class IlluminantPreset:
    """Manages illuminant definitions and color tint profiles."""
    
    PRESETS = {
        'daylight': [1.0, 1.0, 1.0],
        'red_tint': [2.2, 1.0, 0.7],
        'blue_tint': [0.6, 1.0, 1.8],
        'orange_tint': [2.0, 1.0, 0.6],
        'yellow_tint': [1.7, 1.0, 0.8],
    }
    
    @staticmethod
    def get_illuminant(name):
        """Return illuminant RGB values for given name."""
        return IlluminantPreset.PRESETS.get(name)
    
    @staticmethod
    def get_all_presets():
        """Return all available illuminant presets."""
        return IlluminantPreset.PRESETS.copy()
    
    @staticmethod
    def list_names():
        """Return list of available preset names."""
        return list(IlluminantPreset.PRESETS.keys())


class ImageTinter:
    """Applies illuminant tints to images using sRGB color space."""
    
    @staticmethod
    def apply_tint(img, illuminant_rgb, strength=1.0):
        """
        Apply illuminant tint to image.
        strength: 0.0 (no tint) to 1.0 (full tint).
        """
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
        else:
            img_bgr = img
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        illuminant = np.array(illuminant_rgb, dtype=np.float32)
        illuminant = illuminant / illuminant[1]  # Normalize to green channel
        
        illuminant_adjusted = (illuminant - 1.0) * strength + 1.0
        img_tinted = img_rgb * illuminant_adjusted
        img_tinted = np.clip(img_tinted, 0, 255).astype(np.uint8)
        
        return img_tinted


class PreviewGenerator:
    """Generates visualization previews of illuminant tints."""
    
    def __init__(self, output_dir="preview"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, sample_image, strength=0.85):
        """Generate and display tint preview grid."""
        img_bgr = cv2.imread(sample_image)
        if img_bgr is None:
            print(f"Error: Could not load {sample_image}")
            return False
        
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        presets = IlluminantPreset.get_all_presets()
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        for idx, (name, illuminant) in enumerate(presets.items(), 1):
            if idx >= len(axes):
                break
            
            tinted = ImageTinter.apply_tint(img_bgr, illuminant, strength)
            
            preview_path = os.path.join(self.output_dir, f"preview_{name}.jpg")
            cv2.imwrite(preview_path, cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))
            
            axes[idx].imshow(tinted)
            axes[idx].set_title(f'{name}\nRGB: {illuminant}', fontsize=12)
            axes[idx].axis('off')
        
        for idx in range(len(presets) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        preview_fig = os.path.join(self.output_dir, 'tint_comparison.png')
        plt.savefig(preview_fig, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Preview saved: {preview_fig}")
        print(f"Individual previews in: {self.output_dir}/")
        print(f"Current strength: {strength}")
        
        return True


class ZipHandler:
    """Manages zip file extraction and creation."""
    
    @staticmethod
    def extract_all(zip_path, extract_dir):
        """Extract all files from zip archive."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    @staticmethod
    def extract_sample(zip_path, sample_dir):
        """Extract middle image as sample from zip."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = [
                f for f in zip_ref.namelist()
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not file_list:
                return None
            
            sample_file = file_list[len(file_list) // 2]
            zip_ref.extract(sample_file, sample_dir)
            return os.path.join(sample_dir, sample_file)
    
    @staticmethod
    def create_zip(source_dir, output_zip):
        """Create compressed zip from directory."""
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in tqdm(files, desc="Creating zip", leave=False):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                    zipf.write(file_path, arcname)


class ImageBatchProcessor:
    """Processes batch of images with illuminant tints."""
    
    def __init__(self, extract_dir, output_dir):
        self.extract_dir = extract_dir
        self.output_dir = output_dir
        self.image_files = self._find_images()
    
    def _find_images(self):
        """Find all image files in extracted directory."""
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(Path(self.extract_dir).rglob(f'*{ext}'))
            images.extend(Path(self.extract_dir).rglob(f'*{ext.upper()}'))
        return images
    
    def setup_output_dirs(self, tint_names):
        """Create output directories for original and each tint."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
        
        for tint_name in tint_names:
            os.makedirs(os.path.join(self.output_dir, tint_name), exist_ok=True)
    
    def process(self, tint_names, strength=0.85):
        """
        Apply tints to all images.
        Returns count of successfully processed images.
        """
        processed_count = 0
        
        for img_path in tqdm(self.image_files, desc="Processing images"):
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                
                filename = img_path.name
                
                original_path = os.path.join(self.output_dir, "original", filename)
                cv2.imwrite(original_path, img_bgr)
                
                for tint_name in tint_names:
                    illuminant = IlluminantPreset.get_illuminant(tint_name)
                    tinted = ImageTinter.apply_tint(img_bgr, illuminant, strength)
                    
                    tint_path = os.path.join(self.output_dir, tint_name, filename)
                    cv2.imwrite(tint_path, cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))
                
                processed_count += 1
            
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        return processed_count


class TintingPipeline:
    """Orchestrates complete illuminant tinting workflow."""
    
    def __init__(self, input_zip, output_zip):
        self.input_zip = input_zip
        self.output_zip = output_zip
        self.temp_dir = "temp_process"
        self.extract_dir = os.path.join(self.temp_dir, "extracted")
        self.output_dir = os.path.join(self.temp_dir, "tinted")
    
    def extract_and_preview(self):
        """Extract sample and generate preview."""
        sample_dir = "temp_sample"
        os.makedirs(sample_dir, exist_ok=True)
        
        print(f"Extracting sample from {self.input_zip}...")
        sample_image = ZipHandler.extract_sample(self.input_zip, sample_dir)
        
        if not sample_image:
            print("Error: No images found in ZIP")
            return None, None
        
        print(f"Sample extracted: {sample_image}")
        
        generator = PreviewGenerator(output_dir="preview")
        return sample_image, sample_dir
    
    def get_user_strength(self, sample_image, initial_strength=0.85):
        """Interactive prompt for tint strength adjustment."""
        strength = initial_strength
        
        print(f"Default strength: {strength}")
        
        generator = PreviewGenerator(output_dir="preview")
        generator.generate(sample_image, strength=strength)
        
        while True:
            response = input(f"\nCurrent strength: {strength}\nEnter new strength (0.0-1.0) or Enter to continue: ")
            
            if response.strip() == "":
                break
            
            try:
                new_strength = float(response)
                if 0 <= new_strength <= 1.0:
                    strength = new_strength
                    print(f"Regenerating preview with strength {strength}...")
                    generator.generate(sample_image, strength=strength)
                else:
                    print("Enter value between 0.0 and 1.0")
            except ValueError:
                print("Invalid input")
        
        return strength
    
    def process_batch(self, tint_names, strength=0.85):
        """Extract, process, and zip all images."""
        print(f"Extracting images from {self.input_zip}...")
        ZipHandler.extract_all(self.input_zip, self.extract_dir)
        
        processor = ImageBatchProcessor(self.extract_dir, self.output_dir)
        processor.setup_output_dirs(tint_names)
        
        print(f"Processing {len(processor.image_files)} images with {len(tint_names)} tints...")
        processed_count = processor.process(tint_names, strength)
        
        print(f"Processed {processed_count} images")
        
        print(f"Creating {self.output_zip}...")
        ZipHandler.create_zip(self.output_dir, self.output_zip)
        
        zip_size = os.path.getsize(self.output_zip) / (1024**2)
        print(f"Created {self.output_zip} ({zip_size:.1f} MB)")
        
        self.cleanup()
        
        return {
            'processed': processed_count,
            'tints': tint_names,
            'output': self.output_zip,
            'zip_size_mb': zip_size
        }
    
    def cleanup(self):
        """Remove temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists("temp_sample"):
            shutil.rmtree("temp_sample")
        print("Cleaned up temporary files")
    
    def run(self):
        """Execute complete workflow with user interaction."""
        sample_image, sample_dir = self.extract_and_preview()
        
        if not sample_image:
            return False
        
        strength = self.get_user_strength(sample_image)
        shutil.rmtree(sample_dir)
        
        print(f"Batch processing with strength: {strength}")
        response = input("Proceed? (yes/no): ")
        
        if response.lower() != 'yes':
            print("Cancelled")
            self.cleanup()
            return False
        
        tint_names = ['red_tint', 'blue_tint', 'orange_tint', 'yellow_tint']
        result = self.process_batch(tint_names, strength)
        
        print(f"\nCompleted: {result['output']}")
        print(f"Images: {result['processed']} × {len(tint_names) + 1} variants")
        
        return True


if __name__ == "__main__":
    input_zip = "filtered_coco_train.zip"
    output_zip = "tinted_images.zip"
    
    pipeline = TintingPipeline(input_zip, output_zip)
    success = pipeline.run()
    
    if success:
        print("All done")
    else:
        print("Process failed or cancelled")