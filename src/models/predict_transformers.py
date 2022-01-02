import argparse
import json
import logging
import os
import sys
from time import gmtime, strftime

import pandas as pd
import torch
from datasets import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from src.data.utils import load_binary_dataset, PCLTransformersDataset
from src.models.utils import load_fast_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/models/pcla_learnprobadist_roberta_base1e-5_158")
parser.add_argument("--test_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/binary_pcl_tune.tsv")
parser.add_argument("--model_type", type=str, default="roberta")

parser.add_argument("--mcd_rounds", type=int, default=0)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=158)  # roberta-base: .95 = 114, .99 = 158

parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--random_seed", type=int, default=17)


if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    assert args.experiment_dir is not None
    _experiment_dir = os.path.join(args.experiment_dir, f"predict_{strftime('%Y-%m-%d_%H-%M-%S', gmtime())}")
    os.makedirs(_experiment_dir)
    logging.info(f"Saving prediction data to {_experiment_dir}")

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(_experiment_dir, f"predict.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    with open(os.path.join(_experiment_dir, "predict_argparse.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    logging.info("Loading test data...")
    test_df = load_binary_dataset(args.test_path)
    logging.info(f"{test_df.shape[0]} test examples")

    test_fname = args.test_path.split(os.path.sep)[-1]
    test_df.to_csv(os.path.join(_experiment_dir, test_fname), sep="\t", index=False)

    logging.info("Loading tokenizer and model...")
    tokenizer = load_fast_tokenizer(tokenizer_type=args.model_type,
                                    pretrained_name_or_path=args.experiment_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.experiment_dir, return_dict=True).to(DEVICE)

    logging.info("Encoding data...")
    test_enc = tokenizer.batch_encode_plus(test_df["text"].tolist(), return_tensors="pt",
                                           padding="max_length", truncation="only_first", max_length=args.max_length)
    if "binary_label" in test_df.columns:
        # NOTE: always using hard labels here because this is the test scenario
        test_enc["labels"] = torch.tensor(test_df["binary_label"].tolist())
    test_dataset = PCLTransformersDataset(**test_enc)

    logging.info("Starting prediction...")
    if args.mcd_rounds > 0:
        model.train()
    else:
        model.eval()

    num_pred_rounds = args.mcd_rounds if args.mcd_rounds > 0 else 1
    test_probas = []

    with torch.no_grad():
        for idx_round in range(num_pred_rounds):
            curr_test_probas = []

            for _curr_batch in tqdm(DataLoader(test_dataset, batch_size=args.batch_size),
                                    total=((len(test_dataset) + args.batch_size - 1) // args.batch_size)):
                curr_batch = {_k: _v.to(DEVICE) for _k, _v in _curr_batch.items()}
                del curr_batch["labels"]
                res = model(**curr_batch)
                probas = torch.softmax(res["logits"], dim=-1).cpu()
                curr_test_probas.append(probas)

            test_probas.append(torch.cat(curr_test_probas))

    test_probas = torch.stack(test_probas)
    mean_test_probas = test_probas.mean(dim=0)
    if num_pred_rounds > 1:
        sd_test_probas = test_probas.std(dim=0)
    else:
        logging.info("Because only 1 prediction round is used, standard deviation is set to 0...")
        sd_test_probas = torch.zeros_like(mean_test_probas, dtype=torch.float32)

    test_preds = torch.argmax(mean_test_probas, dim=-1).cpu().numpy()
    pred_df = {
        "mean(y=PCL)": mean_test_probas[:, 1].numpy().tolist(),
        "sd(y=PCL)": sd_test_probas[:, 1].numpy().tolist(),
        "pred_label": test_preds.tolist()
    }
    if "binary_label" in test_df.columns:
        test_correct = test_dataset.labels.numpy()
        pred_df["correct_label"] = test_correct.tolist()
        test_metrics = {
            "f1_score": f1_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary'),
            "p_score": precision_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary'),
            "r_score": recall_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary')
        }
        logging.info(f"[test] "
                     f"P={test_metrics['p_score']:.3f}, "
                     f"R={test_metrics['r_score']:.3f}, "
                     f"F1={test_metrics['f1_score']:.3f}")
    else:
        logging.info(f"Skipping evaluation because no correct labels are provided inside 'binary_label' column.")

    pd.DataFrame(pred_df).to_csv(os.path.join(_experiment_dir, "predictions.tsv"), sep="\t", index=False)

    # TODO: visualization?
    # ...


