import os

RANDOM_STATE = 42

###################
### About paths ###
###################

ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "..")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, "train_data")
TEST_DATA_FOLDER = os.path.join(DATA_FOLDER, "val_data")
WEIGHTS_FOLDER = os.path.join(ROOT_FOLDER, "weights")

ALL_DOCUMENTS_CSV = os.path.join(DATA_FOLDER, "all_documents.csv")
TRAIN_CSV = os.path.join(DATA_FOLDER, "train.csv")
VALIDATION_CSV = os.path.join(DATA_FOLDER, "validation.csv")
TEST_CSV = os.path.join(DATA_FOLDER, "test.csv")

###########################
### About preprocessing ###
###########################

TEST_SIZE = 0.2


###########################
### About Models        ###
###########################

MEDCAT_ZIP_FILE = os.path.join(WEIGHTS_FOLDER, "medmen_wstatus_2021_oct.zip")
