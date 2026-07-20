import json
from main import create_output_folders, preprocess_dataset, save_processed_dataset_and_split


def main() -> None:
    create_output_folders()
    dataset, summary = preprocess_dataset()
    train_df, test_df, split_summary = save_processed_dataset_and_split(dataset)
    summary["train_rows"] = int(len(train_df))
    summary["test_rows"] = int(len(test_df))
    summary["split"] = split_summary
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
