import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from tqdm import tqdm

from src.data.utils import load_binary_dataset
from src.visualization.visualize import visualize_bin_predictions

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_dir", type=str, default="pcla_keyword_baseline")
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_train.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_tune.tsv")
# Note: test scenario included as this is a purely-offline baseline (not sending this as submission)
parser.add_argument("--test_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_dev.tsv")

parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--optimized_metric", type=str, default="f1_score",
                    choices=["f1_score", "p_score", "r_score"])

parser.add_argument("--random_seed", type=int, default=17)


if __name__ == "__main__":
    args = parser.parse_args()
    OPTIMIZED_METRIC = args.optimized_metric.lower()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "training.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    with open(os.path.join(args.experiment_dir, "train_argparse.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    logging.info("Loading data...")
    train_df = load_binary_dataset(args.train_path)
    dev_df = load_binary_dataset(args.dev_path)
    test_df = load_binary_dataset(args.test_path)
    logging.info(f"{train_df.shape[0]} train, {dev_df.shape[0]} dev, {test_df.shape[0]} test examples")
    num_classes = len(set(train_df["binary_label"].values))

    # Save the data along with the model in case we need it at a later point
    train_fname = args.train_path.split(os.path.sep)[-1]
    dev_fname = args.dev_path.split(os.path.sep)[-1]
    test_fname = args.test_path.split(os.path.sep)[-1]
    train_df.to_csv(os.path.join(args.experiment_dir, train_fname), sep="\t", index=False)
    dev_df.to_csv(os.path.join(args.experiment_dir, dev_fname), sep="\t", index=False)
    test_df.to_csv(os.path.join(args.experiment_dir, test_fname), sep="\t", index=False)

    KEYWORDS = ["migrant", "women", "vulnerable", "refugee", "homeless",
                "immigrant", "in-need", "disabled", "hopeless", "poor-families"]
    keyword2idx = {kw: i for i, kw in enumerate(KEYWORDS)}

    train_features = np.zeros((train_df.shape[0], len(KEYWORDS)), dtype=np.float32)
    for idx_ex, curr_kw in enumerate(train_df["keyword"].tolist()):
        train_features[idx_ex, keyword2idx[curr_kw]] = 1.0
    train_features = torch.from_numpy(train_features)
    train_labels = torch.tensor(train_df["binary_label"])

    dev_features = np.zeros((dev_df.shape[0], len(KEYWORDS)), dtype=np.float32)
    for idx_ex, curr_kw in enumerate(dev_df["keyword"].tolist()):
        dev_features[idx_ex, keyword2idx[curr_kw]] = 1.0
    dev_features = torch.from_numpy(dev_features)
    dev_labels = torch.tensor(dev_df["binary_label"])

    test_features = np.zeros((test_df.shape[0], len(KEYWORDS)), dtype=np.float32)
    for idx_ex, curr_kw in enumerate(test_df["keyword"].tolist()):
        test_features[idx_ex, keyword2idx[curr_kw]] = 1.0
    test_features = torch.from_numpy(test_features)
    test_labels = torch.tensor(test_df["binary_label"])

    best_dev_metric_value = 0.0
    linear_model = nn.Linear(in_features=len(KEYWORDS), out_features=num_classes, bias=False)
    optimizer = optim.SGD(params=linear_model.parameters(), lr=args.learning_rate)
    # Save just in case the model never reaches F1 > 0 (can happen with dummy baselines)
    torch.save(linear_model.state_dict(), os.path.join(args.experiment_dir, "linear_weights.th"))

    ce_loss = nn.CrossEntropyLoss()
    for idx_epoch in range(args.max_epochs):
        shuffled_indices = torch.randperm(train_features.shape[0], dtype=torch.long)
        num_tr_batches = (train_features.shape[0] + args.batch_size - 1) // args.batch_size
        train_loss = 0.0

        # TRAINING ###
        linear_model.train()
        for idx_batch in tqdm(range(num_tr_batches)):
            curr_indices = shuffled_indices[idx_batch * args.batch_size:
                                            (idx_batch + 1) * args.batch_size]

            logits = linear_model(train_features[curr_indices])
            curr_loss = ce_loss(logits, train_labels[curr_indices])

            train_loss += float(curr_loss)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        num_tr_batches += train_features.shape[0] / args.batch_size
        logging.info(f"[train] loss={train_loss / max(1, num_tr_batches):.3f}")

        # VALIDATION ###
        num_dev_batches = (dev_features.shape[0] + args.batch_size - 1) // args.batch_size
        dev_preds = []
        with torch.no_grad():
            linear_model.eval()
            for idx_batch in tqdm(range(num_dev_batches)):
                logits = linear_model(dev_features[idx_batch * args.batch_size:
                                                   (idx_batch + 1) * args.batch_size])
                preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
                dev_preds.append(preds)

        dev_preds = torch.cat(dev_preds).numpy()

        dev_metrics = {
            "f1_score": f1_score(y_true=dev_labels.numpy(), y_pred=dev_preds, pos_label=1, average='binary'),
            "p_score": precision_score(y_true=dev_labels.numpy(), y_pred=dev_preds, pos_label=1, average='binary'),
            "r_score": recall_score(y_true=dev_labels.numpy(), y_pred=dev_preds, pos_label=1, average='binary')
        }
        logging.info(f"[dev] P={dev_metrics['p_score']:.3f}, "
                     f"R={dev_metrics['r_score']:.3f}, "
                     f"F1={dev_metrics['f1_score']:.3f}")

        if dev_metrics[OPTIMIZED_METRIC] > best_dev_metric_value:
            best_dev_metric_value = dev_metrics[OPTIMIZED_METRIC]
            torch.save(linear_model.state_dict(), os.path.join(args.experiment_dir, "linear_weights.th"))

    logging.info(f"----\n"
                 f"Best validation {OPTIMIZED_METRIC}: {best_dev_metric_value:.3f}")
    linear_model.load_state_dict(torch.load(os.path.join(args.experiment_dir, "linear_weights.th")))

    # Visualize weights
    proba_weights = torch.softmax(linear_model.weight, dim=0).detach().numpy()
    plt.barh(np.arange(len(KEYWORDS)) - 0.45 / 2, proba_weights[0], height=0.45, color="darkorange")
    plt.barh(np.arange(len(KEYWORDS)) + 0.45 / 2, proba_weights[1], height=0.45, color="darkblue")
    plt.legend(["No_PCL", "PCL"])
    plt.yticks(np.arange(len(KEYWORDS)), KEYWORDS)
    plt.tight_layout()
    plt.savefig(os.path.join(args.experiment_dir, "weights.png"))

    # Reobtain dev preds for visualization
    num_dev_batches = (dev_features.shape[0] + args.batch_size - 1) // args.batch_size
    dev_probas = []
    dev_preds = []
    with torch.no_grad():
        linear_model.eval()
        for idx_batch in tqdm(range(num_dev_batches)):
            logits = linear_model(dev_features[idx_batch * args.batch_size:
                                               (idx_batch + 1) * args.batch_size])
            curr_probas = torch.softmax(logits, dim=-1)
            curr_preds = torch.argmax(curr_probas, dim=-1)

            dev_probas.append(curr_probas)
            dev_preds.append(curr_preds)

    dev_probas = torch.cat(dev_probas).numpy()
    dev_preds = torch.cat(dev_preds).numpy()
    dev_labels = dev_labels.numpy()
    logging.info("Saving validation set predictions...")
    visualize_bin_predictions(texts=dev_df["text"].tolist(),
                              preds=dev_preds,
                              correct=dev_labels,
                              mean_pos_probas=dev_probas[:, 1],
                              visualization_save_path=os.path.join(args.experiment_dir, "vis_dev_predictions.html"))
    pd.DataFrame({
        "mean(y=PCL)": dev_probas[:, 1].tolist(),
        "sd(y=PCL)": np.zeros_like(dev_probas[:, 1]).tolist(),
        "pred_label": dev_preds.tolist(),
        "correct_label": test_labels.tolist()
    }).to_csv(os.path.join(args.experiment_dir, "dev_predictions.tsv"), sep="\t", index=False)

    # TEST ###
    num_test_batches = (test_features.shape[0] + args.batch_size - 1) // args.batch_size
    test_preds = []
    test_probas = []
    with torch.no_grad():
        for idx_batch in tqdm(range(num_test_batches)):
            logits = linear_model(test_features[idx_batch * args.batch_size:
                                                (idx_batch + 1) * args.batch_size])
            curr_probas = torch.softmax(logits, dim=-1)
            curr_preds = torch.argmax(curr_probas, dim=-1)

            test_probas.append(curr_probas)
            test_preds.append(curr_preds)

    test_probas = torch.cat(test_probas).numpy()
    test_preds = torch.cat(test_preds).numpy()
    test_labels = test_labels.numpy()
    test_metrics = {
        "f1_score": f1_score(y_true=test_labels, y_pred=test_preds, pos_label=1, average='binary'),
        "p_score": precision_score(y_true=test_labels, y_pred=test_preds, pos_label=1, average='binary'),
        "r_score": recall_score(y_true=test_labels, y_pred=test_preds, pos_label=1, average='binary')
    }
    logging.info(f"[test] P={test_metrics['p_score']:.3f}, "
                 f"R={test_metrics['r_score']:.3f}, "
                 f"F1={test_metrics['f1_score']:.3f}")
    logging.info("Saving test set predictions...")
    visualize_bin_predictions(texts=test_df["text"].tolist(),
                              preds=test_preds,
                              correct=test_labels,
                              mean_pos_probas=test_probas[:, 1],
                              visualization_save_path=os.path.join(args.experiment_dir, "vis_test_predictions.html"))
    pd.DataFrame({
        "mean(y=PCL)": test_probas[:, 1].tolist(),
        "sd(y=PCL)": np.zeros_like(test_probas[:, 1]).tolist(),
        "pred_label": test_preds.tolist(),
        "correct_label": test_labels.tolist()
    }).to_csv(os.path.join(args.experiment_dir, "test_predictions.tsv"), sep="\t", index=False)
