from src.dataset.extract import create_all_docs_csv, create_test_csv
from src.dataset.train_valid_split import create_train_valid_csv

if __name__ == "__main__":
    create_all_docs_csv()
    create_test_csv()
    create_train_valid_csv()
