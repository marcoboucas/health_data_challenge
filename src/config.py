import os

RANDOM_STATE = 42

###################
### About paths ###
###################

ROOT_FOLDER = "./"

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, "train_data")
TEST_DATA_FOLDER = os.path.join(DATA_FOLDER, "val_data")

ALL_DOCUMENTS_CSV = os.path.join(DATA_FOLDER, "all_documents.csv")
TRAIN_CSV = os.path.join(DATA_FOLDER, "train.csv")
VALIDATION_CSV = os.path.join(DATA_FOLDER, "validation.csv")
TEST_CSV = os.path.join(DATA_FOLDER, "test.csv")

###########################
### About preprocessing ###
###########################

TEST_SIZE = 0.2
