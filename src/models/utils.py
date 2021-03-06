import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, XLNetTokenizerFast

KEYWORDS = ["migrant", "women", "vulnerable", "refugee", "homeless",
            "immigrant", "in-need", "disabled", "hopeless", "poor-families"]


def bracketed_representation(tag):
    # Converts a tag (e.g. UPOS, NER, sentiment) into unified scheme: tag -> [TAG]
    return f"[{tag.upper()}]"


NER_TAGS = list(map(bracketed_representation,
                    ["O",
                     "B-ORG", "I-ORG", "E-ORG", "S-ORG",
                     "B-PER", "I-PER", "E-PER", "S-PER",
                     "B-LOC", "I-LOC", "E-LOC", "S-LOC",
                     "B-MISC", "I-MISC", "E-MISC", "S-MISC"]))

NER_SEQ_TAGS = list(map(bracketed_representation,
                        ["ORG", "/ORG", "PER", "/PER", "LOC", "/LOC", "MISC", "/MISC"]))

UPOS_TAGS = list(map(bracketed_representation,
                     ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                      "PUNCT", "SCONJ", "SYM", "VERB", "X"]))
# Penn treebank tags
XPOS_TAGS = list(map(bracketed_representation,
                     ["#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH",
                      "IN", "JJ", "JJR", "JJS", "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP",
                      "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                      "WDT", "WP", "WP$", "WRB", "``"]))

DEPREL_TAGS = list(map(bracketed_representation,
                       ["acl", "acl:relcl", "advcl", "advmod", "advmod:emph", "advmod:lmod", "amod", "appos", "aux", "aux:pass",
                        "case", "cc", "cc:preconj", "ccomp", "clf", "compound", "compound:lvc", "compound:prt", "compound:redup", "compound:svc",
                        "conj", "cop", "csubj", "csubj:pass", "dep", "det", "det:numgov", "det:nummod", "det:poss", "discourse",
                        "dislocated", "expl", "expl:impers", "expl:pass", "expl:pv", "fixed", "flat", "flat:foreign", "flat:name",
                        "goeswith", "iobj", "list", "mark", "nmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:pass", "nummod", "nummod:gov",
                        "obj", "obl", "obl:agent", "obl:arg", "obl:lmod", "obl:tmod", "orphan", "parataxis", "punct", "reparandum",
                        "root", "vocative", "xcomp"]))

# Max 13 entities in training set (50 is way more than enough)
MAX_ENTITIES_IN_DOC = 50
COREF_ENTITY_TAGS = list(map(bracketed_representation,
                             ["O"] + [f"ENTITY{_i}" for _i in range(MAX_ENTITIES_IN_DOC)]))
COREF_SEQ_ENTITY_TAGS = list(map(bracketed_representation, [f"ENTITY{_i}" for _i in range(MAX_ENTITIES_IN_DOC)])) + \
                        list(map(bracketed_representation, [f"/ENTITY{_i}" for _i in range(MAX_ENTITIES_IN_DOC)]))

SENTIMENT_TAGS = list(map(bracketed_representation,
                          ["NEG_SENT", "OBJ_SENT", "POS_SENT", "UNK_SENT"]))
SENTENCE_SENTIMENT_TAGS = list(map(bracketed_representation,
                                   ["NEGATIVE", "NEUTRAL", "POSITIVE"]))


def load_fast_tokenizer(tokenizer_type, pretrained_name_or_path):
    # There is no AutoTokenizerFast??
    assert tokenizer_type in ["bert", "distilbert", "roberta", "xlm-roberta", "xlnet"]

    if tokenizer_type == "bert":
        return BertTokenizerFast.from_pretrained(pretrained_name_or_path)
    elif tokenizer_type == "distilbert":
        return DistilBertTokenizerFast.from_pretrained(pretrained_name_or_path)
    elif tokenizer_type == "roberta":
        return RobertaTokenizerFast.from_pretrained(pretrained_name_or_path)
    elif tokenizer_type == "xlm-roberta":
        return XLMRobertaTokenizerFast.from_pretrained(pretrained_name_or_path)
    elif tokenizer_type == "xlnet":
        return XLNetTokenizerFast.from_pretrained(pretrained_name_or_path)


def optimize_threshold(y_true, y_proba_pos, validated_metric):
    assert validated_metric in ["p_score", "r_score", "f1_score"]
    if validated_metric == "p_score":
        metric_fn = precision_score
    elif validated_metric == "r_score":
        metric_fn = recall_score
    else:
        metric_fn = f1_score

    valid_thresholds = sorted(list(set(y_proba_pos)))
    best_thresh, best_metric_value = None, 0.0
    for curr_thresh in valid_thresholds:
        curr_preds = (y_proba_pos >= curr_thresh).astype(np.int32)
        curr_metric_value = metric_fn(y_true=y_true, y_pred=curr_preds,
                                      pos_label=1, average='binary')

        if curr_metric_value > best_metric_value:
            best_metric_value = curr_metric_value
            best_thresh = curr_thresh

    return best_thresh, best_metric_value
