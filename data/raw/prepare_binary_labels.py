import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv")
parser.add_argument("--target_path", type=str, default="../interim/binary_pcl.tsv")


def convert_label_to_proba(fourway_label):
    # Instead of hard conversion into binary label, do soft conversion, by assuming
    # 0 -> P(y=PCL) = 0.0, 1 -> P(y=PCL) = 0.5, and 2 -> P(y=PCL) = 1.0 and averaging the two judgements
    if fourway_label == 0:
        return [1.0, 0.0]
    elif fourway_label == 1:
        return [0.75, 0.25]
    elif fourway_label == 2:
        return [0.5, 0.5]
    elif fourway_label == 3:
        return [0.25, 0.75]
    elif fourway_label == 4:
        return [0.0, 1.0]
    else:
        raise ValueError(f"Unrecognized label {fourway_label}")


if __name__ == "__main__":
    args = parser.parse_args()
    # First 4 rows contain a comment/disclaimer
    df = pd.read_csv(args.data_path, sep="\t", skiprows=4, header=None)
    df.columns = ["par_id", "art_id", "keyword", "country_code", "text", "label"]

    # Convert to binary label, following grouping in Perez Almendros et al. (2020)
    df["binary_label"] = df["label"].apply(lambda fourway_label: int(fourway_label in {2, 3, 4}))

    # Convert to non-onehot binary label
    df["proba_binary_label"] = df["label"].apply(convert_label_to_proba)
    print(f"Writing converted data to {args.target_path}")
    df.to_csv(args.target_path, sep="\t", index=False)


