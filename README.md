# Breast-cancer-segmentation---computer-vision
Working on DICOM image dataset with BIRADs categorization. Training the model using FasterRCNN algorithm after drawing the bounding boxes of the tumors using annotation files. 

Steps to run the program
1. Sample dataset has been downloaded and is loaded in the DICOM_data folder
2. The annotations files for the tumor is included in the finding annotation file
   1. The categorization of tumor is done in form of BIRADS. BIRADS 1 and 2 are tumorless.
   2. Whereas BIRADS 3,4 and 5 have tumors in them hence has annotation details in the file.
3. Run these files in order
  1. Deidentification
  2. split data
  3. visualize
  4. preprocessing pipeline
  5. Model_run

So far, the DICOM images are hashed, the data is split into training and test based on the BIRADS and masses, converted DICOM into numpy format and stored in the folder named preprocessed_data. Model_run code is used to draw the bounding boxes using annotations to train the model on the region of interest.

[<p align="center"><img src=".github/logo-VinBigData-2020-ngang-blue.png" width="300"></p>](https://vindr.ai/)
# VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography
[<p align="center"><img src=".github/mammo_thumbnail.png" width="400"></p>](https://vindr.ai/datasets/mammo)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Description
This repository provides code used for de-identification and stratification of the VinDr-Mammo dataset, which can be downloaded via [our project on Physionet](https://physionet.org/content/vindr-mammo/1.0.0/) . Python script for visualization of DICOM image is also provided.

# Installation

To install required package via Pip, run 
```bash
pip install -r requirements.txt
```

# De-identification
See the [deidentification.py](deidentification.py) file for more details. 

# Data Stratification
Please refer to the [stratification.py](stratification.py) file and the [split_data.ipynb](split_data.ipynb) notebook.
You may need to change the GLOBAL_PATH and LOCAL_PATH variables in split_data.ipynb to proper paths to the annotations files.
# Visualization
Change the dicom_path variable in the [visualize.py](visualize.py) file to your desired DICOM file for visualization.

```bash
python visualize.py






The project is in progress. Wait for more commits to come to get the final application in form of an end-product. 
