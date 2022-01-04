from transformers import BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, XLNetTokenizerFast

KEYWORDS = ["migrant", "women", "vulnerable", "refugee", "homeless",
            "immigrant", "in-need", "disabled", "hopeless", "poor-families"]
NER_TAGS = ["O", "B-ORG", "I-ORG", "E-ORG", "B-PER", "I-PER", "E-PER", "B-LOC", "I-LOC", "E-LOC"]
UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
             "SCONJ", "SYM", "VERB", "X"]
# Penn treebank tags
XPOS_TAGS = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
             "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
             "VBZ", "WDT", "WP", "WP$", "WRB"]

DEPREL_TAGS = ["acl", "acl:relcl", "advcl", "advmod", "advmod:emph", "advmod:lmod", "amod", "appos", "aux", "aux:pass",
               "case", "cc", "cc:preconj", "ccomp", "clf", "compound", "compound:lvc", "compound:prt", "compound:redup", "compound:svc",
               "conj", "cop", "csubj", "csubj:pass", "dep", "det", "det:numgov", "det:nummod", "det:poss", "discourse",
               "dislocated", "expl", "expl:impers", "expl:pass", "expl:pv", "fixed", "flat", "flat:foreign", "flat:name",
               "goeswith", "iobj", "list", "mark", "nmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:pass", "nummod", "nummod:gov",
               "obj", "obl", "obl:agent", "obl:arg", "obl:lmod", "obl:tmod", "orphan", "parataxis", "punct", "reparandum",
               "root", "vocative", "xcomp"]

# Max 13 entities in training set (50 is way more than enough)
MAX_ENTITIES_IN_DOC = 50
COREF_ENTITY_TAGS = ["O"] + list(map(lambda i: f"[ENTITY{i}]", range(MAX_ENTITIES_IN_DOC)))

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
