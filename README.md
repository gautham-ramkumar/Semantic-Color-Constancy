"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2025-12-11
CS7180 - Advanced Perception
"""
OS Used: Windows(VS Code)
Drive link for dataset: https://northeastern-my.sharepoint.com/personal/kasireddy_y_northeastern_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkasireddy%5Fy%5Fnortheastern%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2Ftinted%5Fimages%2Ezip&parent=%2Fpersonal%2Fkasireddy%5Fy%5Fnortheastern%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ga=1
 
Steps to run data preparation for the project: (Not Recommended,but if needed)
1. Download the Train 2017 Images and Annotations from COCO website
2. Run the cocofilter_saveindices, applyfilter_rezip and apply tint files in order
3. You will have the imagefiles in this format:
### Dataset Structure

The dataset is organized as follows:

```text
tinted_images/
├── original/       # Original unmodified images
├── red_tint/       # Red-tinted variants
├── blue_tint/      # Blue-tinted variants
├── orange_tint/    # Orange-tinted variants
└── yellow_tint/    # Yellow-tinted variants
 
 
Steps to run the algorithm:
1. Download the codes from the gradescope and put them in a folder (let's call it Adv_perc).
2. Download the Original and Tinted Dataset from the Drive file below or from the report into the Adv_perc folder.

https://northeastern-my.sharepoint.com/personal/kasireddy_y_northeastern_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkasireddy%5Fy%5Fnortheastern%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2Ftinted%5Fimages%2Ezip&parent=%2Fpersonal%2Fkasireddy%5Fy%5Fnortheastern%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ga=1

3. Next step is to install the dependencies
	pip install numpy opencv-python pyyaml tqdm ultralytics
	pip install torch
4. Now we need to build the priors you can use any dataset you want but our method is currently optimized for the provided dataset in the drive use the full original folder to train the priors. Run
	python build_color_priors.py `
  		--data "C:\path\to\Adv_perc\original" `
  		--model "C:\path\to\Adv_perc\yolov8s-seg.pt" `
  		--out "C:\path\to\Adv_perc\data\color_priors.yaml"
5. Now that we built the priors we'll test our color correction algorithm. To Run the color correction we have two ways one is to run on a folder and the other is to run it on a single image.
a) To run it on a folder Run
	python apply_color_correction_v2.py `
  		--data "C:\path\to\Adv_perc\test_images" `
  		--priors "C:\path\to\Adv_perc\data\color_priors.yaml"
  		--model "C:\path\to\Adv_perc\yolov8s-seg.pt" `
  		--conf 0.5 `
  		--out_dir "C:\path\to\corrected_outputs"
	The Input and output can be whatever you want.
b) To run it on a single image Run
	python apply_color_correction_v2.py `
  		--image "C:\path\to\Adv_perc\test_images\img_1.jpg" `
  		--priors "C:\path\to\Adv_perc\data\color_priors.yaml"
  		--model "C:\path\to\Adv_perc\yolov8s-seg.pt" `
  		--conf 0.5 `
  		--out_dir "C:\path\to\corrected_outputs"



Steps to run the analysis code:
1. To Run this code you need to have the corrected images in one file and ground truth or original images in the other.
2. Then Run the file - finaltests.py using the following command

python finaltests.py 
	--data "C:\path\to\corrected_outputs" ` 
	--priors "C:\path\to\Adv_perc\data\color_priors.yaml" `
	--gt_data "\path\to\ground_truth"




Steps to run the Extension:
 
1. Run the File - inverseapproach.py with the following command  line arguments
 
python apply_color_correction_inverse.py `
    --image "C:\path\to\test_image.jpg" `
    --priors "C:\path\to\Adv_perc\data\color_priors.yaml" `
    --method "inverse_semantic" `
    --out_dir "C:\path\to\outputs"
 
Arguments:
- `--image` - Input image path
- `--priors` - Color priors YAML file
- `--method` - Either `semantic` or `inverse_semantic`
- `--out_dir` - Output directory
