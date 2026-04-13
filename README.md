1) Install
pip install -r requirements.txt

requirements.txt currently lists torch, torchvision, and numpy, matplotlib 
2) Expected dataset format

Your training/eval code expects this folder structure:

dataset/
  train/
    accept/
    reject/
  val/
    accept/
    reject/
  test/
    accept/
    reject/

That is because model.py builds datasets from data_dir/train, data_dir/val, and data_dir/test using ImageFolder, and it expects class folders named exactly accept and reject. Alphabetical ordering means accept=0 and reject=1 is the expected mapping.

3) Create dataset from Excel + image folders

Edit the config at the top of bowtie_data_split.py:

ROOT = folder containing the .xlsx files and image folders
OUT = output dataset folder
COL_IMG_ID = image id column
COL_AR = column containing A / R
HEADER_ROW = first data row

The script normalizes image IDs, maps A -> accept and R -> reject, resolves matching image folders, and creates a stratified 70/15/15 train/val/test split. It also writes reports like summary.txt and missing_images.tsv.

Run it:

python bowtie_data_split.py
4) Run inference on one image

Use predict.py for a single image:

python predict.py --model runs/bowtie_model/best_model.pt --image path/to/image.jpg --img-size 448

It loads best_model.pt, reads the saved threshold from the checkpoint, and prints Prob reject plus the final Prediction.

5) Run inference on a whole folder

Use run_model.py for batch prediction on a directory:

python run_model.py --model runs/bowtie_model/best_model.pt --input-dir dataset/test --img-size 448 --output-file predictions.csv

What it does:

walks all images under --input-dir
predicts each image
writes CSV with columns: image_path, prob_reject, prediction

Important: run_model.py is batch inference, not true labeled evaluation. If you point it at dataset/test, it will predict every test image, but it does not compare predictions against ground truth or compute metrics by itself.

6) If you want actual test metrics

model.py is the training script. It trains on train, validates on val, saves best_model.pt, then evaluates on the labeled test split and writes:

history.json
test_metrics.json

Typical run:

python model.py --data-dir dataset --out-dir runs/bowtie_model --epochs 20 --batch-size 16 --img-size 448
7) Other files

model.py
Main training script. Uses EfficientNet-V2-S, builds train/eval transforms, trains the classifier, saves best_model.pt, and exports final test metrics.

gradcam.py
Visual explanation tool for one image. It loads the checkpoint, finds the last conv layer, generates a Grad-CAM heatmap, and saves a figure showing original image, heatmap, and overlay.

Example:

python gradcam.py --model runs/bowtie_model/best_model.pt --image path/to/image.jpg --img-size 448 --out gradcam_result.png

agr.py
This is your augmentation script. It is not evaluation. It adds extra augmented image copies, by default only inside dataset/train, using light transforms like rotation, brightness/contrast, shift, scale, and optional flips. Default is AUGS_PER_IMAGE = 4.

Run it:

python agr.py
Short practical workflow
# 1) build dataset
python bowtie_data_split.py

# 2) optionally augment training set
python agr.py

# 3) train model and get best_model.pt + test_metrics.json
python model.py --data-dir dataset --out-dir runs/bowtie_model

# 4) single-image inference
python predict.py --model runs/bowtie_model/best_model.pt --image some_image.jpg

# 5) folder inference
python run_model.py --model runs/bowtie_model/best_model.pt --input-dir dataset/test --output-file predictions.csv

# 6) visual explanation
python gradcam.py --model runs/bowtie_model/best_model.pt --image some_image.j
