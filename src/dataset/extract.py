import os

import pandas as pd

from src import config


def create_all_docs_csv() -> None:
    """
    Create a csv file with all the docs
    """
    # Create new `pandas` methods which use `tqdm` progress
    classification_subfolders = ["concept", "ast", "rel"]
    doc_list = []
    for dirname in os.listdir(config.RAW_DATA_FOLDER):
        folder_path = os.path.join(config.RAW_DATA_FOLDER, dirname)
        doc_list.extend(
            [
                {
                    "name": filename,
                    "path": os.path.join(folder_path, "txt", filename),
                    **{
                        c: os.path.join(folder_path, c, filename) for c in classification_subfolders
                    },
                }
                for filename in os.listdir(os.path.join(folder_path, "txt"))
            ]
        )
    df = pd.DataFrame(doc_list)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df.to_csv(config.ALL_DOCUMENTS_CSV)


def create_test_csv() -> None:
    """
    Create a csv file with all the docs
    """
    # Create new `pandas` methods which use `tqdm` progress
    doc_list = []
    folder_path = config.TEST_DATA_FOLDER
    doc_list = [
        {
            "name": filename,
            "path": os.path.join(folder_path, "txt", filename),
        }
        for filename in os.listdir(os.path.join(folder_path, "txt"))
    ]
    df = pd.DataFrame(doc_list)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df.to_csv(config.TEST_CSV)


if __name__ == "__main__":
    create_all_docs_csv()
    create_test_csv()
