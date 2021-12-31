import argparse
import json
import logging
import os
import sys
from time import time, gmtime, strftime

import torch
from sklearn.metrics import f1_score, precision_score
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.data.utils import load_binary_dataset, PCLTransformersDataset, log_and_maybe_print
from src.models.utils import load_fast_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_train.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_tune.tsv")

parser.add_argument("--model_type", type=str, default="roberta")
parser.add_argument("--pretrained_name_or_path", type=str, default="roberta-base")

parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=158)  # roberta-base: .95 = 114, .99 = 158
parser.add_argument("--eval_every_n_examples", type=int, default=3000)
parser.add_argument("--early_stopping_tolerance", type=int, default=5)
parser.add_argument("--optimized_metric", type=str, default="loss",
                    choices=["loss", "f1_score", "p_score", "r_score"])

parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--random_seed", type=int, default=17)


if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    DEV_BATCH_SIZE = args.batch_size * 2

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    OPTIMIZED_METRIC = args.optimized_metric.lower()
    if args.experiment_dir is None:
        args.experiment_dir = f"{strftime('%Y-%m-%d_%H-%M-%S', gmtime())}"

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
        log_and_maybe_print(f"|{k:30s}|{v_str:50s}|")

    log_and_maybe_print("Loading data...")
    train_df = load_binary_dataset(args.train_path)
    dev_df = load_binary_dataset(args.dev_path)
    log_and_maybe_print(f"{train_df.shape[0]} train, {dev_df.shape[0]} dev examples")

    # Save the data along with the model in case we need it at a later point
    train_fname = args.train_path.split(os.path.sep)[-1]
    dev_fname = args.dev_path.split(os.path.sep)[-1]
    train_df.to_csv(os.path.join(args.experiment_dir, train_fname), sep="\t", index=False)
    dev_df.to_csv(os.path.join(args.experiment_dir, dev_fname), sep="\t", index=False)

    log_and_maybe_print("Loading tokenizer and model...")

    tokenizer = load_fast_tokenizer(tokenizer_type=args.model_type,
                                    pretrained_name_or_path=args.pretrained_name_or_path)
    tokenizer.save_pretrained(args.experiment_dir)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", return_dict=True).to(DEVICE)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    log_and_maybe_print("Encoding data...")
    train_enc = tokenizer.batch_encode_plus(train_df["text"].tolist(), return_tensors="pt",
                                            padding="max_length", truncation="only_first", max_length=args.max_length)
    train_enc["labels"] = torch.tensor(train_df["binary_label"].tolist())
    train_dataset = PCLTransformersDataset(**train_enc)

    dev_enc = tokenizer.batch_encode_plus(dev_df["text"].tolist(), return_tensors="pt",
                                          padding="max_length", truncation="only_first", max_length=args.max_length)
    dev_enc["labels"] = torch.tensor(dev_df["binary_label"].tolist())
    dev_dataset = PCLTransformersDataset(**dev_enc)

    stop_training = False
    no_increase = 0
    # Minimize loss, maximize anything else (P/R/F1)
    if OPTIMIZED_METRIC == "loss":
        best_dev_metric_value = float("inf")

        def is_better(_curr, _best):
            return _curr < _best
    else:
        best_dev_metric_value = 0.0

        def is_better(_curr, _best):
            return _curr > _best

    log_and_maybe_print("Starting training...")
    ts = time()
    for idx_epoch in range(args.max_epochs):
        train_loss, num_tr_batches = 0.0, 0
        log_and_maybe_print(f"Epoch #{idx_epoch}...")
        shuffled_indices = torch.randperm(len(train_dataset))

        num_train_subsets = (len(train_dataset) + args.eval_every_n_examples - 1) // args.eval_every_n_examples
        for idx_subset in range(num_train_subsets):
            log_and_maybe_print(f"Subset #{idx_subset}...")
            curr_indices = shuffled_indices[idx_subset * args.eval_every_n_examples:
                                            (idx_subset + 1) * args.eval_every_n_examples]
            curr_train_subset = Subset(train_dataset, curr_indices)

            # TRAINING ###
            model.train()
            for _curr_batch in tqdm(DataLoader(curr_train_subset, batch_size=args.batch_size),
                                    total=((len(curr_train_subset) + args.batch_size - 1) // args.batch_size)):
                curr_batch = {_k: _v.to(DEVICE) for _k, _v in _curr_batch.items()}

                loss = model(**curr_batch)["loss"]
                train_loss += float(loss)

                loss.backward()
                optimizer.step()

            num_tr_batches += len(curr_train_subset) / args.batch_size
            log_and_maybe_print(f"[train] loss={train_loss / max(1, num_tr_batches):.3f}")
            if args.eval_every_n_examples / len(curr_train_subset) > 3:
                log_and_maybe_print(f"Skipping validation because training subset was small "
                                    f"({len(curr_train_subset)} < 1/3 * {args.eval_every_n_examples})")
                continue

            # VALIDATION ###
            dev_loss = 0.0
            dev_preds = []
            with torch.no_grad():
                model.eval()
                for _curr_batch in tqdm(DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE),
                                        total=((len(dev_dataset) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE)):
                    curr_batch = {_k: _v.to(DEVICE) for _k, _v in _curr_batch.items()}

                    res = model(**curr_batch)
                    dev_loss += float(res["loss"])
                    probas = torch.softmax(res["logits"], dim=-1)
                    preds = torch.argmax(probas, dim=-1).cpu()

                    dev_preds.append(preds)

            num_dev_batches = len(dev_dataset) / DEV_BATCH_SIZE
            dev_loss /= num_dev_batches

            dev_preds = torch.cat(dev_preds).numpy()
            dev_correct = dev_dataset.labels.numpy()

            dev_metrics = {
                "loss": dev_loss,
                "f1_score": f1_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary'),
                "p_score": precision_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary'),
                "r_score": precision_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary')
            }
            log_and_maybe_print(f"[dev] loss={dev_loss:.3f}, "
                                f"P={dev_metrics['p_score']:.3f}, "
                                f"R={dev_metrics['r_score']:.3f}, "
                                f"F1={dev_metrics['f1_score']:.3f}")

            if is_better(_curr=dev_metrics[OPTIMIZED_METRIC], _best=best_dev_metric_value):
                best_dev_metric_value = dev_metrics[OPTIMIZED_METRIC]
                no_increase = 0

                log_and_maybe_print(f"Improved validation {OPTIMIZED_METRIC}, saving model state...")
                model.save_pretrained(args.experiment_dir)
            else:
                no_increase += 1

            if no_increase == args.early_stopping_tolerance:
                log_and_maybe_print(f"Stopping training after validation {OPTIMIZED_METRIC} did not improve for "
                                    f"{args.early_stopping_tolerance} checks...")
                stop_training = True
                break

        if stop_training:
            break

    te = time()
    log_and_maybe_print(f"Training took {te - ts:.3f}s /\n"
                        f"best validation {OPTIMIZED_METRIC}: {best_dev_metric_value:.3f}")

