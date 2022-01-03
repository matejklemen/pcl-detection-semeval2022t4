import argparse
import json
import logging
import os
import sys
from time import time, strftime, gmtime
from typing import List

import stanza
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from src.data.utils import load_binary_dataset, PCLTransformersDataset
from src.models.utils import NER_TAGS

parser = argparse.ArgumentParser()
parser.add_argument("--use_label_probas", action="store_true",
                    help="Whether to use soft labels (label probas) instead of one-hot encoded labels")

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


class NERRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            ne_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        # FIXME: set to fixed value because I CBA fixing this
        output_attentions = False
        output_hidden_states = False
        return_dict = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_main = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_ner = self.roberta(
            ne_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = (outputs_main[0] + outputs_ner[0]) * 0.5  # TODO: learnable weights
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs_main[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def process_stanza(stanza_tokenizer: stanza.Pipeline, hf_tokenizer, examples: List[str], max_length):
    tokens_or_words, tags = [], []

    STANZA_BATCH_SIZE = 1024
    num_stanza_batches = (len(examples) + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE
    for idx_batch in tqdm(range(num_stanza_batches)):
        curr_examples = examples[idx_batch * STANZA_BATCH_SIZE: (idx_batch + 1) * STANZA_BATCH_SIZE]
        processed = stanza_tokenizer("\n\n".join(curr_examples))
        for curr_sent in processed.sentences:
            sent_toks_or_words, sent_tags = [], []
            for curr_tok in curr_sent.tokens:
                sent_toks_or_words.append(curr_tok.words[0].text)
                sent_tags.append(curr_tok.ner)

            tokens_or_words.append(sent_toks_or_words)
            tags.append(sent_tags)

    encoded = hf_tokenizer.batch_encode_plus(tokens_or_words, is_split_into_words=True, return_tensors="pt",
                                             padding="max_length", truncation="only_first", max_length=max_length)
    encoded["ne_ids"] = encoded["input_ids"].clone()
    for idx_example in range(len(tokens_or_words)):
        for position, (curr_id, curr_word_id) in enumerate(zip(encoded["input_ids"][idx_example],
                                                               encoded.word_ids(batch_index=idx_example))):
            if curr_word_id is not None:
                encoded["ne_ids"][idx_example, position] = \
                    tokenizer.encode(tags[idx_example][curr_word_id], add_special_tokens=False)[0]

    return encoded


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
        logging.info(f"|{k:30s}|{v_str:50s}|")

    logging.info("Loading data...")
    train_df = load_binary_dataset(args.train_path)
    dev_df = load_binary_dataset(args.dev_path)
    logging.info(f"{train_df.shape[0]} train, {dev_df.shape[0]} dev examples")

    # Save the data along with the model in case we need it at a later point
    train_fname = args.train_path.split(os.path.sep)[-1]
    dev_fname = args.dev_path.split(os.path.sep)[-1]
    train_df.to_csv(os.path.join(args.experiment_dir, train_fname), sep="\t", index=False)
    dev_df.to_csv(os.path.join(args.experiment_dir, dev_fname), sep="\t", index=False)

    nlp = stanza.Pipeline("en", processors={"tokenize": "combined", "ner": "conll03"}, tokenize_no_ssplit=True,
                          # FIXME: (hack) provide invalid package so stanza doesn't go loading every processor
                          package="joze",
                          use_gpu=(not args.use_cpu))

    model = NERRobertaForSequenceClassification.from_pretrained(args.pretrained_name_or_path, return_dict=True).to(DEVICE)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    tokenizer.add_special_tokens({
        "additional_special_tokens": list(map(lambda s: f"[{s.upper()}]", NER_TAGS))
    })
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(args.experiment_dir)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    train_enc = process_stanza(stanza_tokenizer=nlp,
                               hf_tokenizer=tokenizer,
                               examples=train_df["text"].tolist(),
                               max_length=args.max_length)
    label_probas = torch.zeros((train_df.shape[0], 2), dtype=torch.float32)
    if args.use_label_probas:
        label_probas = torch.tensor(train_df["proba_binary_label"].tolist(), dtype=torch.float32)
    else:
        label_probas[torch.arange(train_df.shape[0]), train_df["binary_label"].tolist()] = 1.0

    train_enc["labels"] = label_probas
    train_dataset = PCLTransformersDataset(**train_enc)

    dev_enc = process_stanza(stanza_tokenizer=nlp,
                             hf_tokenizer=tokenizer,
                             examples=dev_df["text"].tolist(),
                             max_length=args.max_length)
    # Note: we do not want to change dev labels (0.5/0.5 would get turned from 1 to 0)
    label_probas = torch.zeros((dev_df.shape[0], 2), dtype=torch.float32)
    label_probas[torch.arange(dev_df.shape[0]), dev_df["binary_label"].tolist()] = 1.0

    dev_enc["labels"] = label_probas
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

    ce_loss = CrossEntropyLoss()
    logging.info("Starting training...")
    ts = time()
    for idx_epoch in range(args.max_epochs):
        train_loss, num_tr_batches = 0.0, 0
        logging.info(f"Epoch #{idx_epoch}...")
        shuffled_indices = torch.randperm(len(train_dataset))

        num_train_subsets = (len(train_dataset) + args.eval_every_n_examples - 1) // args.eval_every_n_examples
        for idx_subset in range(num_train_subsets):
            logging.info(f"Subset #{idx_subset}...")
            curr_indices = shuffled_indices[idx_subset * args.eval_every_n_examples:
                                            (idx_subset + 1) * args.eval_every_n_examples]
            curr_train_subset = Subset(train_dataset, curr_indices)

            # TRAINING ###
            model.train()
            for _curr_batch in tqdm(DataLoader(curr_train_subset, batch_size=args.batch_size),
                                    total=((len(curr_train_subset) + args.batch_size - 1) // args.batch_size)):
                curr_batch = {_k: _v.to(DEVICE) for _k, _v in _curr_batch.items()}

                logits = model(**curr_batch)["logits"]
                loss = ce_loss(logits, curr_batch["labels"])

                train_loss += float(loss)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            num_tr_batches += len(curr_train_subset) / args.batch_size
            logging.info(f"[train] loss={train_loss / max(1, num_tr_batches):.3f}")
            if args.eval_every_n_examples / len(curr_train_subset) > 3:
                logging.info(f"Skipping validation because training subset was small "
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
                    loss = ce_loss(res["logits"], curr_batch["labels"])
                    dev_loss += float(loss)
                    probas = torch.softmax(res["logits"], dim=-1)
                    preds = torch.argmax(probas, dim=-1).cpu()

                    dev_preds.append(preds)

            num_dev_batches = len(dev_dataset) / DEV_BATCH_SIZE
            dev_loss /= num_dev_batches

            dev_preds = torch.cat(dev_preds).numpy()
            dev_correct = torch.argmax(dev_dataset.labels, dim=-1).numpy()

            dev_metrics = {
                "loss": dev_loss,
                "f1_score": f1_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary'),
                "p_score": precision_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary'),
                "r_score": recall_score(y_true=dev_correct, y_pred=dev_preds, pos_label=1, average='binary')
            }
            logging.info(f"[dev] loss={dev_loss:.3f}, "
                         f"P={dev_metrics['p_score']:.3f}, "
                         f"R={dev_metrics['r_score']:.3f}, "
                         f"F1={dev_metrics['f1_score']:.3f}")

            if is_better(_curr=dev_metrics[OPTIMIZED_METRIC], _best=best_dev_metric_value):
                best_dev_metric_value = dev_metrics[OPTIMIZED_METRIC]
                no_increase = 0

                logging.info(f"Improved validation {OPTIMIZED_METRIC}, saving model state...")
                model.save_pretrained(args.experiment_dir)
            else:
                no_increase += 1

            if no_increase == args.early_stopping_tolerance:
                logging.info(f"Stopping training after validation {OPTIMIZED_METRIC} did not improve for "
                             f"{args.early_stopping_tolerance} checks...")
                stop_training = True
                break

        if stop_training:
            break

    te = time()
    logging.info(f"Training took {te - ts:.3f}s /\n"
                 f"best validation {OPTIMIZED_METRIC}: {best_dev_metric_value:.3f}")
