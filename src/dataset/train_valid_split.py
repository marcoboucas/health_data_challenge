import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


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
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    train.to_csv(config.TRAIN_CSV)
    valid.to_csv(config.VALIDATION_CSV)


if __name__ == "__main__":
    create_train_valid_csv()
