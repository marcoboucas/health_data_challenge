import os

import pandas as pd

from src import config


def create_all_docs_csv() -> None:
    """
    Create a csv file with all the docs
    """
    doc_list = []
    for dirname in os.listdir(config.RAW_DATA_FOLDER):
        folder_path = os.path.join(config.RAW_DATA_FOLDER, dirname)
        doc_list.extend(
            [
                {
                    "name": filename.split(".")[0],
                    "path": os.path.join(folder_path, "txt", filename),
                    "concept": os.path.join(
                        folder_path, "concept", f"{filename.split('.')[0]}.con"
                    ),
                    "rel": os.path.join(folder_path, "rel", f"{filename.split('.')[0]}.rel"),
                    "ast": os.path.join(folder_path, "ast", f"{filename.split('.')[0]}.ast"),
                }
                for filename in os.listdir(os.path.join(folder_path, "txt"))
                if filename[0] != "."
            ]
        )
    df = pd.DataFrame(doc_list)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df.to_csv(config.ALL_DOCUMENTS_CSV)


def create_test_csv() -> None:
    """
    Create a csv file with all the docs
    """
    doc_list = []
    folder_path = config.TEST_DATA_FOLDER
    doc_list = [
        {
            "name": filename,
            "path": os.path.join(folder_path, "txt", filename),
        }
        for filename in os.listdir(os.path.join(folder_path, "txt"))
        if filename[0] != "."
    ]
    df = pd.DataFrame(doc_list)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df.to_csv(config.TEST_CSV)


if __name__ == "__main__":
    create_all_docs_csv()
    create_test_csv()
