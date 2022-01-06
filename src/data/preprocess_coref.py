"""
    Extracts tokens and their coreference information and adds the information as two separate columns.
    This is done because neuralcoref is not well maintained (it is being integrated directly into spacy),
    so it might not work everywhere.
    This can be run somewhere where there are no problems with spacy + neuralcoref, and the model is not dependent on
    the packages.

    Tested with:
        Python 3.8.10
        spacy==2.1.0
        neuralcoref==4.0
"""

import argparse

import neuralcoref
import spacy
from tqdm import tqdm

from src.data.utils import load_binary_dataset
from src.models.utils import bracketed_representation, MAX_ENTITIES_IN_DOC

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/matej/Documents/multiview-pcl-detection/data/processed/80_10_10/binary_pcl_dev.tsv")
parser.add_argument("--target_path", type=str,
                    default="binary_pcl_dev_augmented.tsv")

if __name__ == "__main__":
    args = parser.parse_args()
    df = load_binary_dataset(args.data_path)
    examples = df["text"].tolist()

    spacy_model = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(spacy_model)

    tokens_or_words, tags = [], []
    for idx_ex in tqdm(range(len(examples))):
        doc = spacy_model(examples[idx_ex])

        curr_tokens, curr_tags = [], []
        for w in doc:
            curr_tokens.append(w.text)
            clusters_of_w = w._.coref_clusters

            if w._.in_coref:
                # Current simplification: assign to first cluster (could also do random - watch out for spans!)
                first_cluster = clusters_of_w[0]
                if first_cluster.i >= MAX_ENTITIES_IN_DOC:
                    curr_tags.append(bracketed_representation("O"))
                else:
                    # TODO: combine NER with coref tags? (maybe even BIOES?)
                    curr_tags.append(bracketed_representation(f"ENTITY{first_cluster.i}"))
            else:
                curr_tags.append(bracketed_representation("O"))

        tokens_or_words.append(curr_tokens)
        tags.append(curr_tags)

    df["spacy_tokens"] = tokens_or_words
    df["spacy_coref"] = tags
    df.to_csv(args.target_path, sep="\t", index=False)






