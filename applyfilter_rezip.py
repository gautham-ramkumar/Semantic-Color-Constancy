"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2024-06-15
COCO dataset Filtering and Rezipping Images - Data Preparation Part 2
CS7180 - Advanced Perception
Main reason they are in 2 scripts instead of one is to check on the images and make sure we don't have any bad filters.
"""

import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm


class KeepListLoader:
    """Load the list of image filenames to keep."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.files = set()
    
    def load(self):
        """Read keep list from file."""
        with open(self.filepath, 'r') as f:
            self.files = set(line.strip() for line in f if line.strip())
        return self.files


class ImageFilter:
    """Find which images to copy and which to skip."""
    
    def __init__(self, source_dir, keep_files):
        self.source_dir = source_dir
        self.keep_files = keep_files
    
    def get_images_to_process(self):
        """Compare source images with keep list."""
        source_images = [img.name for img in Path(self.source_dir).glob("*.jpg")]
        source_set = set(source_images)
        
        to_copy = self.keep_files & source_set
        to_skip = source_set - self.keep_files
        missing = self.keep_files - source_set
        
        return {
            'total': len(source_images),
            'to_copy': to_copy,
            'to_skip': to_skip,
            'missing': missing
        }


class ImageCopier:
    """Copy filtered images to output folder."""
    
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def copy(self, filenames, dry_run=False):
        """Copy images from source to output."""
        stats = {'copied': 0, 'skipped': 0, 'failed': 0}
        
        for filename in tqdm(sorted(filenames), desc="Copying"):
            src = os.path.join(self.source_dir, filename)
            dst = os.path.join(self.output_dir, filename)
            
            if dry_run:
                stats['copied'] += 1
                continue
            
            try:
                if os.path.exists(dst):
                    stats['skipped'] += 1
                else:
                    shutil.copy2(src, dst)
                    stats['copied'] += 1
            except Exception as e:
                print(f"Error: {filename} - {e}")
                stats['failed'] += 1
        
        return stats


class ZipBuilder:
    """Create zip archive from images."""
    
    def __init__(self, folder, output_path):
        self.folder = folder
        self.output_path = output_path
    
    def build(self, dry_run=False):
        """Compress images into zip."""
        if dry_run:
            print(f"Would create: {self.output_path}")
            return True
        
        try:
            images = list(Path(self.folder).glob("*.jpg"))
            
            with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for img in tqdm(images, desc="Zipping"):
                    zf.write(img, arcname=img.name)
            
            return True
        except Exception as e:
            print(f"Zip error: {e}")
            return False


class FilterPipeline:
    """Main workflow for filtering and zipping images."""
    
    def __init__(self, source_dir, keep_list, output_dir, zip_name):
        self.source_dir = os.path.expanduser(source_dir)
        self.keep_list = os.path.expanduser(keep_list)
        self.output_dir = os.path.expanduser(output_dir)
        self.zip_path = os.path.join(os.path.dirname(self.output_dir) or '.', zip_name)
    
    def validate(self):
        """Check that inputs exist."""
        if not os.path.exists(self.source_dir):
            print(f"Source dir not found: {self.source_dir}")
            return False
        
        if not os.path.exists(self.keep_list):
            print(f"Keep list not found: {self.keep_list}")
            return False
        
        return True
    
    def run(self, dry_run=True, delete_after=False):
        """Execute filtering and zipping."""
        print("Starting filter pipeline...\n")
        
        if not self.validate():
            return None
        
        print("Loading keep list...")
        loader = KeepListLoader(self.keep_list)
        keep = loader.load()
        print(f"  {len(keep):,} files to keep\n")
        
        print("Scanning source...")
        filter_obj = ImageFilter(self.source_dir, keep)
        plan = filter_obj.get_images_to_process()
        
        print(f"  Total images: {plan['total']:,}")
        print(f"  Will copy: {len(plan['to_copy']):,}")
        print(f"  Will skip: {len(plan['to_skip']):,}")
        if plan['missing']:
            print(f"  Not found: {len(plan['missing']):,}")
        print()
        
        if not dry_run:
            confirm = input("Type 'YES' to continue: ")
            if confirm != "YES":
                print("Cancelled")
                return None
        
        print("Copying images...")
        copier = ImageCopier(self.source_dir, self.output_dir)
        copy_result = copier.copy(plan['to_copy'], dry_run=dry_run)
        
        if not dry_run:
            print(f"  Copied: {copy_result['copied']:,}")
            if copy_result['skipped']:
                print(f"  Skipped: {copy_result['skipped']:,}")
            if copy_result['failed']:
                print(f"  Failed: {copy_result['failed']:,}")
        print()
        
        print("Creating zip...")
        zb = ZipBuilder(self.output_dir, self.zip_path)
        if not zb.build(dry_run=dry_run):
            return None
        
        if not dry_run:
            size_mb = os.path.getsize(self.zip_path) / (1024**2)
            print(f"  Created: {self.zip_path} ({size_mb:.1f} MB)\n")
        
        if delete_after and not dry_run:
            confirm = input(f"Delete source folder? Type 'DELETE': ")
            if confirm == "DELETE":
                try:
                    shutil.rmtree(self.source_dir)
                    print("Source deleted")
                except Exception as e:
                    print(f"Delete error: {e}")
        
        print("\nDone!")
        
        return {
            'output': self.output_dir,
            'zip': self.zip_path,
            'stats': copy_result
        }


if __name__ == "__main__":
    source = "C:/Users/yogak/Downloads/train2017/train2017"
    keep_file = "analysis_results/images_to_KEEP.txt"
    output = "filtered_coco_images"
    zip_file = "filtered_coco_train.zip"
    
    pipeline = FilterPipeline(source, keep_file, output, zip_file)
    result = pipeline.run(dry_run=False, delete_after=True)
    
    if result:
        print("Success!")
    else:
        print("Failed or cancelled")