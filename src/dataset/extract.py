"""Dataset extraction and structuration."""
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def create_all_annotated_docs_csv() -> None:
    """
    Create a csv file with all the docs that are annotated
    """
    doc_list = []
    for dirname in os.listdir(config.RAW_DATA_FOLDER):
        folder_path = os.path.join(config.RAW_DATA_FOLDER, dirname)
        doc_list.extend(
            [
                {
                    "name": filename.split(".")[0],
                    "txt": os.path.join(folder_path, "txt", filename),
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
    df = pd.DataFrame(doc_list).applymap(lambda x: x.replace("\\", "/"))
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
            "txt": os.path.join(folder_path, "txt", filename),
        }
        for filename in os.listdir(os.path.join(folder_path, "txt"))
        if filename[0] != "."
    ]

    df = pd.DataFrame(doc_list).applymap(lambda x: x.replace("\\", "/"))
    df = df.sample(frac=1, random_state=config.RANDOM_STATE)
    df.to_csv(config.TEST_CSV, index=False)


def create_train_valid_csv() -> None:
    """
    Create and save a train and a valid file
    """
    df_tot = pd.read_csv(config.ALL_DOCUMENTS_CSV, index_col=0)
    train, valid = train_test_split(
        df_tot,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    train.to_csv(config.TRAIN_CSV, index=False)
    valid.to_csv(config.VAL_CSV, index=False)


def create_dataset_folders():
    """Create the datasets folders."""
    # Train set
    for file_path, data_folder in zip(
        [config.TRAIN_CSV, config.VAL_CSV], [config.TRAIN_DATA_FOLDER, config.VAL_DATA_FOLDER]
    ):
        df: pd.DataFrame = pd.read_csv(file_path)

        for folder in ["txt", "ast", "concept", "rel"]:
            os.makedirs(os.path.join(data_folder, folder), exist_ok=True)
        for row in df.itertuples():
            shutil.copyfile(row.txt, os.path.join(data_folder, "txt", f"{row.name}.txt"))
            shutil.copyfile(row.concept, os.path.join(data_folder, "concept", f"{row.name}.con"))
            shutil.copyfile(row.rel, os.path.join(data_folder, "rel", f"{row.name}.rel"))
            shutil.copyfile(row.ast, os.path.join(data_folder, "ast", f"{row.name}.ast"))

        for folder in ["txt", "ast", "concept", "rel"]:
            # pylint: disable=cell-var-from-loop
            df[folder] = df[folder].apply(
                lambda x: os.path.join(data_folder, folder, os.path.basename(x)).replace("\\", "/")
            )
        df.to_csv(file_path, index=False)


def create_datasets_csv_files():
    """Create the datasets."""
    create_all_annotated_docs_csv()
    create_test_csv()
    create_train_valid_csv()


if __name__ == "__main__":
    create_dataset_folders()
