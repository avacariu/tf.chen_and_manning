from __future__ import print_function

import argparse

from collections import defaultdict
import tensorflow as tf
import utils
import model


def train(args):
    with tf.Graph().as_default(), tf.Session() as session:
        words, pos, rels = utils.extract_vocab(args.train_file)

        vocab = defaultdict(utils.return1)    # we want to default to UNK not PAD
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
                        tags, relations, session,
                        activation=args.activation,
                        batch_size=args.batch_size,
                        l2_weight=args.l2,
                        learning_rate=args.learning_rate)

        init = tf.global_variables_initializer()
        session.run(init)

        m.train(trees, epochs=args.epochs,
                dropout_keep_prob=args.dropout_keep_prob)

        m.save_to(args.save_to)


def test(args):
    with tf.Graph().as_default(), tf.Session() as session:
        m = model.Model.load_from(args.model, session)

        with open(args.test_file) as f:
            sentences, trees = utils.read_conll(f, m.vocab, m.tags, m.relations)

        m.parse(sentences, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", default=100000, type=int)
    train_parser.add_argument("--train-file", default="data/UD_English/en-ud-train.conllu")
    train_parser.add_argument("--batch-size", default=512, type=int)
    train_parser.add_argument("--activation", default="cubed", choices=["cubed", "relu", "tanh"])
    train_parser.add_argument("--l2", default=1e-8, type=float)
    train_parser.add_argument("--embedding-size", default=50, type=int)
    train_parser.add_argument("--hidden-layer-size", default=200, type=int)
    train_parser.add_argument("--learning-rate", default=0.01, type=float)
    train_parser.add_argument("--dropout-keep-prob", default=0.5, type=float)
    train_parser.add_argument("--save-to", required=True,
                              help="Save variable here. Will also create a SAVE_TO.params file which stores the model parameters.")
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", required=True,
                             help="This should match the --save-to path. There must also be a MODEL.params file available.")
    test_parser.add_argument("--output", required=True)
    test_parser.add_argument("--test-file", default="data/UD_English/en-ud-test.conllu")
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
