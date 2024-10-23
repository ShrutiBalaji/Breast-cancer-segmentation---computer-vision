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

The project is in progress. Wait for more commits to come to get the final application in form of an end-product. 
