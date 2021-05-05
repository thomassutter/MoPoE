import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

# digit_text = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq, alphabet, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def create_text_from_label_mnist(len_seq, label, alphabet):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples;

