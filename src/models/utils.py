from transformers import BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, T5TokenizerFast, MT5TokenizerFast, ElectraTokenizerFast, XLNetTokenizerFast


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
