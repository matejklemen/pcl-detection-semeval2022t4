import argparse
import logging
import sys

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, RobertaTokenizerFast

from src.data.utils import load_binary_dataset, PCLTransformersDataset
from src.models.utils import load_fast_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="roberta")
parser.add_argument("--pretrained_name_or_path", type=str, default="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
parser.add_argument("--test_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/80_10_10/binary_pcl_tune.tsv")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=256)  # NOTE: > 158 because using sequence pairs now

parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--random_seed", type=int, default=17)

if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    BATCH_SIZE = args.batch_size

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout)]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    tokenizer = load_fast_tokenizer(args.model_type, args.pretrained_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path)
    model.eval()

    KEYWORD = "patronizing"
    PROMPTS = {
        "migrant": [f"This text is not {KEYWORD} towards migrants.",
                    f"This text is {KEYWORD} towards migrants."],
        "women": [f"This text is not {KEYWORD} towards women.",
                  f"This text is {KEYWORD} towards women."],
        "vulnerable": [f"This text is not {KEYWORD} towards vulnerable groups.",
                       f"This text is {KEYWORD} towards vulnerable groups."],
        "refugee": [f"This text is not {KEYWORD} towards refugees.",
                    f"This text is {KEYWORD} towards refugees."],
        "homeless": [f"This text is not {KEYWORD} towards homeless people.",
                     f"This text is {KEYWORD} towards homeless people."],
        "immigrant": [f"This text is not {KEYWORD} towards immigrants.",
                      f"This text is {KEYWORD} towards immigrants."],
        "in-need": [f"This text is not {KEYWORD} towards people in need.",
                    f"This text is {KEYWORD} towards people in need."],
        "disabled": [f"This text is not {KEYWORD} towards disabled people.",
                     f"This text is {KEYWORD} towards disabled people."],
        "hopeless": [f"This text is not {KEYWORD} towards hopeless people.",
                     f"This text is {KEYWORD} towards hopeless people."],
        "poor-families": [f"This text is not {KEYWORD} towards poor families.",
                          f"This text is {KEYWORD} towards poor families."],
        "other": [f"This text is not {KEYWORD}.",
                  f"This text is {KEYWORD}."]
    }
    PROMPTS_PER_EXAMPLE = 2

    test_df = load_binary_dataset(args.test_path)

    prepared_test_examples = []
    for curr_kw, curr_text in test_df[["keyword", "text"]].values:
        curr_prompts = PROMPTS.get(curr_kw, "other")
        prepared_test_examples.append((curr_text, curr_prompts[0]))  # not PCL
        prepared_test_examples.append((curr_text, curr_prompts[1]))  # PCL

    test_enc = tokenizer.batch_encode_plus(prepared_test_examples, return_tensors="pt",
                                           padding="max_length", truncation="only_first",
                                           max_length=args.max_length)
    test_dataset = PCLTransformersDataset(**test_enc)
    test_logits = []
    with torch.no_grad():
        for _curr_batch in tqdm(DataLoader(test_dataset, batch_size=args.batch_size),
                                total=((len(test_dataset) + args.batch_size - 1) // args.batch_size)):
            curr_batch = {_k: _v.to(DEVICE) for _k, _v in _curr_batch.items()}
            res = model(**curr_batch)
            test_logits.append(res["logits"].cpu())

        test_logits = torch.cat(test_logits)
        # TODO: currently this is hardcoded for two prompts (i.e. first and second prompt predictions are intertwined)
        # Take the entailment probability for contradictory prompt and affirmative prompt
        test_logits = torch.cat((
            test_logits[::2, 0].unsqueeze(1),
            test_logits[1::2, 0].unsqueeze(1)
        ), dim=1)
        test_probas = torch.softmax(test_logits, dim=-1)

        test_preds = torch.argmax(test_probas, dim=-1)

    if "binary_label" in test_df.columns:
        test_correct = test_df["binary_label"].values
        test_preds = test_preds.numpy()

        test_metrics = {
            "f1_score": f1_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary'),
            "p_score": precision_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary'),
            "r_score": recall_score(y_true=test_correct, y_pred=test_preds, pos_label=1, average='binary')
        }
        logging.info(f"[test] P={test_metrics['p_score']:.3f}, "
                     f"R={test_metrics['r_score']:.3f}, "
                     f"F1={test_metrics['f1_score']:.3f}")













