"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2024-06-15
COCO dataset analysis and filtering based on object diversity criteria. - Data Preparation Part 1
CS7180 - Advanced Perception
"""

import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import os
import csv


class ObjectCategories:
    """Manages object category definitions and classification."""
    
    def __init__(self):
        self.strong_anchors = {
            'refrigerator', 'oven', 'microwave', 'sink', 'toilet'
        }
        self.medium_anchors = {
            'tv', 'laptop', 'keyboard', 'mouse', 'remote', 'toaster'
        }
        self.context_objects = {
            'person', 'chair', 'couch', 'bed', 'dining table',
            'bottle', 'cup', 'bowl', 'book', 'vase', 
            'cell phone', 'clock', 'potted plant'
        }
        self.all_indoor = self.strong_anchors | self.medium_anchors | self.context_objects
    
    def get_category_type(self, category_name):
        """Returns category type: strong, medium, or context."""
        if category_name in self.strong_anchors:
            return 'strong'
        elif category_name in self.medium_anchors:
            return 'medium'
        return 'context'
    
    def is_indoor_object(self, category_name):
        """Check if category is an indoor object."""
        return category_name in self.all_indoor


class COCODataLoader:
    """Loads COCO JSON and provides access to annotations and metadata."""
    
    def __init__(self, annotations_file):
        self.annotations_file = annotations_file
        self.data = None
        self.cat_id_to_name = {}
        self.img_id_to_info = {}
    
    def load(self):
        """Load COCO JSON and build lookup tables."""
        with open(self.annotations_file, 'r') as f:
            self.data = json.load(f)
        
        self.cat_id_to_name = {
            cat['id']: cat['name'] 
            for cat in self.data['categories']
        }
        
        self.img_id_to_info = {
            img['id']: img 
            for img in self.data['images']
        }
        
        return {
            'total_images': len(self.data['images']),
            'total_annotations': len(self.data['annotations']),
            'total_categories': len(self.data['categories'])
        }
    
    def get_image_info(self, img_id):
        """Retrieve image metadata by ID."""
        return self.img_id_to_info.get(img_id)
    
    def get_category_name(self, cat_id):
        """Retrieve category name by ID."""
        return self.cat_id_to_name.get(cat_id)
    
    def get_annotations(self):
        """Return all annotations."""
        return self.data['annotations']
    
    def get_images(self):
        """Return all image metadata."""
        return self.data['images']


class AnnotationProcessor:
    """Processes annotations and groups objects by image and category."""
    
    def __init__(self, coco_loader, categories):
        self.loader = coco_loader
        self.categories = categories
    
    def process(self, min_area_ratio=0.01):
        """Process annotations and group objects by image and category."""
        image_objects = defaultdict(lambda: defaultdict(list))
        
        for annotation in tqdm(self.loader.get_annotations(), desc="Processing annotations"):
            img_id = annotation['image_id']
            cat_id = annotation['category_id']
            cat_name = self.loader.get_category_name(cat_id)
            
            if not self.categories.is_indoor_object(cat_name):
                continue
            
            img_info = self.loader.get_image_info(img_id)
            if not img_info:
                continue
            
            img_area = img_info['width'] * img_info['height']
            obj_area = annotation.get('area', 0)
            
            if obj_area / img_area < min_area_ratio:
                continue
            
            image_objects[img_id][cat_name].append({
                'area': obj_area,
                'area_ratio': obj_area / img_area,
                'bbox': annotation['bbox']
            })
        
        return image_objects


class DiversityAnalyzer:
    """Evaluates image diversity based on object composition."""
    
    def __init__(self, categories):
        self.categories = categories
    
    def evaluate_image(self, img_id, filename, objects_by_cat,
                      min_unique_categories=2, min_total_objects=2,
                      require_strong_anchor=True):
        """Evaluate single image for diversity criteria."""
        unique_categories = list(objects_by_cat.keys())
        num_unique_cats = len(unique_categories)
        total_objects = sum(len(objs) for objs in objects_by_cat.values())
        
        has_strong_anchor = any(
            cat in self.categories.strong_anchors 
            for cat in unique_categories
        )
        
        object_details = [
            {
                'category': cat_name,
                'count': len(objs),
                'type': self.categories.get_category_type(cat_name)
            }
            for cat_name, objs in objects_by_cat.items()
        ]
        
        diversity_score = (
            num_unique_cats * 2 +
            total_objects +
            (10 if has_strong_anchor else 0)
        )
        
        fail_reasons = []
        if num_unique_cats < min_unique_categories:
            fail_reasons.append(f'only_{num_unique_cats}_categories')
        if total_objects < min_total_objects:
            fail_reasons.append(f'only_{total_objects}_objects')
        if require_strong_anchor and not has_strong_anchor:
            fail_reasons.append('no_strong_anchor')
        
        status = 'failed' if fail_reasons else 'passed'
        
        return {
            'image_id': img_id,
            'filename': filename,
            'status': status,
            'num_unique_categories': num_unique_cats,
            'total_objects': total_objects,
            'has_strong_anchor': has_strong_anchor,
            'diversity_score': diversity_score,
            'categories': unique_categories,
            'object_details': object_details,
            'fail_reasons': fail_reasons if fail_reasons else None
        }
    
    def evaluate_all(self, images, image_objects, **criteria):
        """Evaluate all images and return classification results."""
        results = {'passed': [], 'failed': []}
        
        for image in tqdm(images, desc="Analyzing images"):
            img_id = image['id']
            objects_by_cat = image_objects.get(img_id, {})
            
            analysis = self.evaluate_image(
                img_id, image['file_name'], objects_by_cat, **criteria
            )
            
            results[analysis['status']].append(analysis)
        
        results['passed'].sort(key=lambda x: x['diversity_score'], reverse=True)
        return results


class ResultsWriter:
    """Writes analysis results to multiple output formats."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write_summary(self, summary):
        """Write analysis summary JSON."""
        filepath = os.path.join(self.output_dir, "analysis_summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        return filepath
    
    def write_detailed_results(self, results):
        """Write detailed image analysis JSON."""
        filepath = os.path.join(self.output_dir, "detailed_analysis.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        return filepath
    
    def write_category_statistics(self, category_analysis):
        """Write category statistics JSON sorted by frequency."""
        filepath = os.path.join(self.output_dir, "category_statistics.json")
        sorted_data = dict(sorted(
            category_analysis.items(),
            key=lambda x: -x[1]['images_containing']
        ))
        with open(filepath, 'w') as f:
            json.dump(sorted_data, f, indent=2)
        return filepath
    
    def write_keep_list(self, results):
        """Write filenames of images to keep."""
        filepath = os.path.join(self.output_dir, "images_to_KEEP.txt")
        with open(filepath, 'w') as f:
            for item in results['passed']:
                f.write(item['filename'] + '\n')
        return filepath
    
    def write_top_diverse_images(self, results, top_n=1000):
        """Write top diverse images with scores."""
        filepath = os.path.join(self.output_dir, "top_diverse_images.txt")
        with open(filepath, 'w') as f:
            for item in results['passed'][:top_n]:
                f.write(
                    f"{item['filename']}\t{item['diversity_score']}\t"
                    f"{item['num_unique_categories']} cats\t"
                    f"{item['total_objects']} objs\n"
                )
        return filepath
    
    def write_delete_list(self, results):
        """Write filenames of images to delete."""
        filepath = os.path.join(self.output_dir, "images_to_DELETE.txt")
        with open(filepath, 'w') as f:
            for item in results['failed']:
                f.write(item['filename'] + '\n')
        return filepath
    
    def write_csv_report(self, results):
        """Write comprehensive CSV report."""
        filepath = os.path.join(self.output_dir, "analysis_report.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'status', 'unique_cats', 'total_objs',
                'diversity_score', 'has_strong_anchor', 'categories'
            ])
            
            for status in ['passed', 'failed']:
                for item in results[status]:
                    writer.writerow([
                        item['filename'],
                        item['status'],
                        item['num_unique_categories'],
                        item['total_objects'],
                        item['diversity_score'],
                        'Yes' if item['has_strong_anchor'] else 'No',
                        ', '.join(item['categories']) if item['categories'] else 'None'
                    ])
        return filepath


class StatisticsBuilder:
    """Builds summary statistics from analysis results."""
    
    def __init__(self, categories):
        self.categories = categories
    
    def build_summary(self, total_images, results, criteria):
        """Build overall summary statistics."""
        passed = len(results['passed'])
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': total_images,
            'passed': passed,
            'failed': len(results['failed']),
            'pass_rate': f"{(passed/total_images*100):.2f}%",
            'criteria': criteria,
            'object_types': {
                'strong_anchors': list(self.categories.strong_anchors),
                'medium_anchors': list(self.categories.medium_anchors),
                'context_objects': list(self.categories.context_objects)
            }
        }
    
    def build_category_stats(self, results):
        """Build per-category statistics."""
        category_stats = defaultdict(int)
        category_counts = defaultdict(list)
        
        for img_data in results['passed']:
            for obj_detail in img_data['object_details']:
                cat = obj_detail['category']
                count = obj_detail['count']
                category_stats[cat] += 1
                category_counts[cat].append(count)
        
        category_analysis = {}
        for cat, count in category_stats.items():
            counts_list = category_counts[cat]
            category_analysis[cat] = {
                'images_containing': count,
                'avg_count_per_image': sum(counts_list) / len(counts_list),
                'max_count': max(counts_list),
                'type': self.categories.get_category_type(cat)
            }
        
        return category_analysis


class COCOAnalysisPipeline:
    """Orchestrates complete analysis workflow."""
    
    def __init__(self, annotations_file, output_dir="analysis_results"):
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.categories = ObjectCategories()
        self.loader = COCODataLoader(annotations_file)
        self.writer = ResultsWriter(output_dir)
        self.stats_builder = StatisticsBuilder(self.categories)
    
    def run(self, min_unique_categories=2, min_total_objects=2,
            min_area_ratio=0.01, require_strong_anchor=True):
        """Execute complete analysis pipeline."""
        
        print("Loading COCO annotations...")
        load_stats = self.loader.load()
        print(f"Images: {load_stats['total_images']:,}, "
              f"Annotations: {load_stats['total_annotations']:,}")
        
        print("Processing annotations...")
        processor = AnnotationProcessor(self.loader, self.categories)
        image_objects = processor.process(min_area_ratio)
        print(f"Found {len(image_objects):,} images with indoor objects")
        
        print("Analyzing diversity...")
        analyzer = DiversityAnalyzer(self.categories)
        results = analyzer.evaluate_all(
            self.loader.get_images(),
            image_objects,
            min_unique_categories=min_unique_categories,
            min_total_objects=min_total_objects,
            require_strong_anchor=require_strong_anchor
        )
        
        print("Building statistics...")
        criteria = {
            'min_unique_categories': min_unique_categories,
            'min_total_objects': min_total_objects,
            'min_area_ratio': min_area_ratio,
            'require_strong_anchor': require_strong_anchor
        }
        summary = self.stats_builder.build_summary(
            load_stats['total_images'], results, criteria
        )
        category_analysis = self.stats_builder.build_category_stats(results)
        
        print("Writing results...")
        self.writer.write_summary(summary)
        self.writer.write_detailed_results(results)
        self.writer.write_category_statistics(category_analysis)
        self.writer.write_keep_list(results)
        self.writer.write_top_diverse_images(results)
        self.writer.write_delete_list(results)
        self.writer.write_csv_report(results)
        
        self._print_summary(summary, category_analysis, results)
        
        return results, summary
    
    def _print_summary(self, summary, category_analysis, results):
        """Print analysis summary to console."""
        total = summary['total_images']
        passed = summary['passed']
        
        print(f"\nTotal images analyzed: {total:,}")
        print(f"Passed: {passed:,} ({passed/total*100:.1f}%)")
        print(f"Failed: {summary['failed']:,}")
        
        print("\nTop objects in passed images:")
        for i, (cat, stats) in enumerate(
            sorted(category_analysis.items(),
                   key=lambda x: -x[1]['images_containing'])[:10], 1
        ):
            print(f"  {i}. {cat}: {stats['images_containing']:,} images")
        
        print("\nTop 5 diverse images:")
        for i, item in enumerate(results['passed'][:5], 1):
            print(f"  {i}. {item['filename']} (score: {item['diversity_score']})")


if __name__ == "__main__":
    annotations_file = "instances_train2017.json"
    output_dir = "analysis_results"
    
    pipeline = COCOAnalysisPipeline(annotations_file, output_dir)
    results, summary = pipeline.run(
        min_unique_categories=2,
        min_total_objects=2,
        min_area_ratio=0.01,
        require_strong_anchor=True
    )