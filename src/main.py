from __future__ import print_function

import argparse

from collections import defaultdict
import tensorflow as tf
import utils
import model

import random


def main(args):
    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(1234)
        random.seed(1234)

        words, pos, rels = utils.extract_vocab(args.train_file)

        vocab = defaultdict(lambda: 1)    # we want to default to UNK not PAD
        vocab.update({word: i+2 for i, word in enumerate(words)})
        vocab["*PAD*"] = 0
        vocab["*UNK*"] = 1

        tags = {word: i+1 for i, word in enumerate(pos)}
        tags['*PAD*'] = 0

        relations = {word: i+1 for i, word in enumerate(rels)}
        relations['*PAD*'] = 0

        with open(args.train_file) as f:
            sentences, trees = utils.read_conll(f, vocab, tags, relations, True)

        m = model.Model(args.embedding_size, args.hidden_layer_size, vocab,
                        tags, relations,
                        activation=args.activation,
                        batch_size=args.batch_size,
                        l2_weight=args.l2,
                        learning_rate=args.learning_rate)

        init = tf.global_variables_initializer()
        session.run(init)

        m.train(trees, session, epochs=args.epochs,
                dropout_keep_prob=args.dropout_keep_prob)

        with open(args.test_file) as f:
            sentences, trees = utils.read_conll(f, vocab, tags, relations)

        m.parse(sentences, args.output, session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100000, type=int)
    parser.add_argument("--train-file", default="data/UD_English/en-ud-train.conllu")
    parser.add_argument("--test-file", default="data/UD_English/en-ud-test.conllu")
    parser.add_argument("--output", default="output.conllu")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--activation", default="cubed", choices=["cubed", "relu", "tanh"])
    parser.add_argument("--l2", default=1e-8, type=float)
    parser.add_argument("--embedding-size", default=50, type=int)
    parser.add_argument("--hidden-layer-size", default=200, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--dropout-keep-prob", default=0.5, type=float)

    args = parser.parse_args()

    main(args)
