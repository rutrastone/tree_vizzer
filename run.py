"""
quick visualization of language model attention weights as dependency trees
"""

import argparse

import numpy as np
import spacy
import torch

from spacy import displacy
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from utils import *

def run(args):

    nlp = spacy.load("xx_ent_wiki_sm")
    doc = nlp(args.sentence)
    tokenizer = BertTokenizer.from_pretrained(args.model,
                                              do_lower_case=False)
    model = BertModel.from_pretrained(args.model,
                                      output_hidden_states=True,
                                      output_attentions=True)

    untokenized_sent = [token.text for token in doc]
    to_tokenize = "[CLS] " + " ".join(untokenized_sent) + " [SEP]"
    tokenized_sent = tokenizer.tokenize(to_tokenize)
    mapping = match_tokenization(tokenized_sent,
                                 untokenized_sent)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        layers = []
        model_out = model(tokens_tensor)[3]
    for i, _ in enumerate(model_out):
        heads = []
        layer = model_out[i].squeeze()
        for j in range(layer.shape[0]):
            summed_incoming = sum_attentions(layer[j],
                                             untokenized_sent,
                                             mapping)
            averaged_outgoing = average_subword_vectors(summed_incoming,
                                                        untokenized_sent,
                                                        mapping)
            heads.append(averaged_outgoing)
        layers.append(torch.stack(heads))
    layers = torch.stack(layers)
    n_layers, n_heads, n_outgoing, n_incoming = layers.size()
    sent_tensor = torch.reshape(layers, (n_layers, n_heads,
                                         n_incoming, n_incoming))
    if args.attn_dist == "max":
        pred_edges = run_max(sent_tensor[args.layer-1][args.head-1])
    elif args.attn_dist == "mst":
        pred_edges = run_cle(sent_tensor[args.layer-1][args.head-1])
    elif args.attn_dist == "js":
        pred_edges = run_js(sent_tensor[args.layer-1][args.head-1])
    else:
        raise ValueError(f"{args.attn_dist} not recognized, has to be in: max, mst, js.")

    input_dict = {"words": [], "arcs": []}
    options = {"bg": "white", "color": "black", "font": "Source Sans Pro"}
    for i, word in enumerate(untokenized_sent):
        j = pred_edges[i]
        input_dict["words"].append({"text": word, "tag": "_"})
        if j == -1:
            continue
        elif j > i:
            arc = {"start": i, "end": j, "label": "_", "dir": "left"}
            input_dict["arcs"].append(arc)
        else:
            arc = {"start": i, "end": j, "label": "_", "dir": "right"}
            input_dict["arcs"].append(arc)

    displacy.serve(input_dict, style="dep", manual=True, options=options, minify=True)

PARSER = argparse.ArgumentParser(description='Perform RSA on a UD treebank.')
PARSER.add_argument('--sentence', metavar='S', type=str,
                    help='Sentence to visualize.')
PARSER.add_argument('--model', metavar='M', type=str,
                    default='bert-base-multilingual-cased',
                    help='HuggingFace modelstring, \
                    e.g.: bert-base-multilingual-cased.')
PARSER.add_argument('--layer', metavar='L', type=int,
                    help='Index of model layer.')
PARSER.add_argument('--head', metavar='H', type=int,
                    help='Index of attention head.')
PARSER.add_argument('--attn_dist', metavar='A', type=str,
                    help='Choice of distance metric: max, mst or js.')

ARGS = PARSER.parse_args()

run(ARGS)
