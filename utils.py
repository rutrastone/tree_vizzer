"""
Random utils.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from allennlp.nn import chu_liu_edmonds as cle
from scipy.spatial.distance import jensenshannon

def average_subword_vectors(
        layer: torch.Tensor,
        untokenized_sent: List[str],
        mapping: Dict[str, int]) -> torch.Tensor:

    """

    Stolen from:
    https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/data.py#L398

    """
    device = "cpu" if layer.get_device() == -1 else "cuda:0"
    avg_layer = []
    for i in range(len(untokenized_sent)):
        to_avg = layer[mapping[i][0]:mapping[i][-1]+1, :]
        avg = np.average(to_avg.cpu().numpy(), axis=0)
        avg_layer.append(avg)
    avg_layer = torch.tensor(avg_layer).to(device)
    assert avg_layer.shape[0] == len(untokenized_sent)

    return avg_layer

def sum_attentions(
        matrix: torch.Tensor,
        untokenized_sent: List[str],
        mapping: Dict[str, int]) -> torch.Tensor:

    device = "cpu" if matrix.get_device() == -1 else "cuda:0"
    matrix = matrix.T
    summed_matrix = []
    for i in range(len(untokenized_sent)):
        to_sum = matrix[mapping[i][0]:mapping[i][-1]+1, :]
        #print(to_max.cpu(), "to_max")
        sum = np.sum(to_sum.cpu().detach().numpy(), axis=0)
        summed_matrix.append(sum)

    summed_matrix = torch.tensor(summed_matrix).squeeze().to(device).T

    return summed_matrix

def match_tokenization(
        tokenized_sent: List[str],
        untokenized_sent: List[str]) -> Dict[str, int]:

    """

    Also stolen from:
    https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/data.py

    Aligns tokenized and untokenized sentence given subwords '##' prefixed
    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, '##', computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
    tokenized_sent: a list of strings describing a subword-tokenized sentence
    untokenized_sent: a list of strings describing a sentence, no subword tok.

    Returns:
    A dictionary of type {int: list(int)} mapping each untokenized sentence
    index to a list of subword-tokenized sentence indices
    """

    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
           tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and
               tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1

    return mapping

def run_cle(matrix):

    matrix = matrix.cpu().numpy()
    n = matrix.shape[0]
    scores = cle.decode_mst(matrix, length=n, has_labels=False)

    return scores[0]

def run_max(matrix):

    matrix = matrix.cpu().numpy()
    n = matrix.shape[0]
    heads = []

    for vec in matrix:
        heads.append(np.argmax(vec))

    return np.array(heads)

def run_js(matrix):

    matrix = matrix.cpu().numpy()
    n = matrix.shape[0]
    heads = []

    for i in range(n):
        distances = np.zeros(n)
        for j in range(n):
            if i != j:
                distances[j] = 1 - jensenshannon(matrix[i], matrix[j])
        heads.append(np.argmax(distances))

    return np.array(heads)
