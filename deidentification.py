import logging
from pydicom.dataset import FileDataset
import hashlib
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# load keeping tags
TAGS_DF = pd.read_csv("./tags.csv")

def deidentify(dicom: FileDataset) -> FileDataset:
    logging.info("Starting de-identification process")
    
    # Hash UIDs
    for uid_field in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
        if hasattr(dicom, uid_field):
            original_uid = getattr(dicom, uid_field)
            masked_uid = hash_fn(original_uid)
            setattr(dicom, uid_field, masked_uid)
            logging.info(f"Masked {uid_field}")
        else:
            logging.warning(f"{uid_field} not found in DICOM file")
    
    # Hash or remove other identifying fields
    for field in ['PatientID', 'PatientName', 'AccessionNumber']:
        if hasattr(dicom, field):
            setattr(dicom, field, "ANONYMOUS")
            logging.info(f"Anonymized {field}")
    
    # Remove non-essential tags
    dicom_keys = list(dicom.keys())
    removed_count = 0
    for k in dicom_keys:
        if k not in TAGS_DF.Decimal.values:
            dicom.pop(k, None)
            removed_count += 1
    logging.info(f"Removed {removed_count} non-essential tags")
    
    return dicom

def hash_fn(text: str) -> str:
    hash_obj = hashlib.sha256(text.encode())
    return hash_obj.hexdigest()



