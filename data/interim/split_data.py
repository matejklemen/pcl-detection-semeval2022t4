import argparse
import os.path

from src.data.utils import load_binary_dataset, train_dev_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="binary_pcl.tsv")
parser.add_argument("--target_dir", type=str, default="../processed")

if __name__ == "__main__":
    args = parser.parse_args()
    df = load_binary_dataset(args.data_path)
    print(f"Loaded dataset with {df.shape[0]} examples from '{args.data_path}'")
    split = train_dev_test_split(df)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    for split_name in ["train", "dev", "test"]:
        print(f"'{split_name}': {split[split_name].shape[0]} examples")
        split[split_name].to_csv(os.path.join(args.target_dir, f"binary_pcl_{split_name}.tsv"), sep="\t", index=False)


