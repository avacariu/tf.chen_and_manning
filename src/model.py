from __future__ import print_function

import sys
import random
import tensorflow as tf

from datatypes import Configuration, TransitionVector
from oracle import Oracle


activation_functions = {
    "cubed": lambda x: x**3,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
}


n_w = 18
n_t = 18
n_l = 12


class Model:

    def __init__(self, d, depth, vocab, tags, relations, activation="cubed",
                 batch_size=64, l2_weight=1e-8, learning_rate=0.01):

        self.vocab = vocab
        self.tags = tags
        self.relations = relations
        self.batch_size = batch_size

        self.transition_vector = TransitionVector(relations)
        num_transitions = len(self.transition_vector)

        self.input_words = tf.placeholder(tf.int32, shape=(None, n_w))
        self.input_tags = tf.placeholder(tf.int32, shape=(None, n_t))
        self.input_labels = tf.placeholder(tf.int32, shape=(None, n_l))
        self.expected = tf.placeholder(tf.int32, shape=(None))
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.W_w = tf.Variable(tf.random_uniform([d*n_w, depth],
                                                 minval=-0.1,
                                                 maxval=0.1))
        self.W_t = tf.Variable(tf.random_uniform([d*n_t, depth],
                                                 minval=-0.1,
                                                 maxval=0.1))
        self.W_l = tf.Variable(tf.random_uniform([d*n_l, depth],
                                                 minval=-0.1,
                                                 maxval=0.1))
        self.b = tf.Variable(tf.random_uniform([depth],
                                               minval=-0.1,
                                               maxval=0.1))
        self.W2 = tf.Variable(tf.random_uniform([depth, num_transitions],
                                                minval=-0.1,
                                                maxval=0.1))

        self.word_embedding = tf.Variable(tf.random_uniform([len(vocab), d],
                                                            minval=-0.01,
                                                            maxval=0.01))
        self.tag_embedding = tf.Variable(tf.random_uniform([len(tags), d],
                                                           minval=-0.01,
                                                           maxval=0.01))
        self.label_embedding = tf.Variable(tf.random_uniform([len(relations), d],
                                                             minval=-0.01,
                                                             maxval=0.01))
        self.trans_embedding = tf.diag(tf.fill([num_transitions],
                                               tf.constant(1, tf.float32)))

        words_embed = tf.nn.embedding_lookup(self.word_embedding,
                                             self.input_words)
        tags_embed = tf.nn.embedding_lookup(self.tag_embedding,
                                            self.input_tags)
        labels_embed = tf.nn.embedding_lookup(self.label_embedding,
                                              self.input_labels)
        trans_embed = tf.nn.embedding_lookup(self.trans_embedding,
                                             self.expected)

        # embedding_lookup gets us a matrix of shape (18, 100), but we need a
        # vector, so we need to flatten that matrix
        x_w = tf.reshape(words_embed, shape=[-1, d*n_w])
        x_t = tf.reshape(tags_embed, shape=[-1, d*n_t])
        x_l = tf.reshape(labels_embed, shape=[-1, d*n_l])

        h_in = (tf.matmul(x_w, self.W_w) +
                tf.matmul(x_t, self.W_t) +
                tf.matmul(x_l, self.W_l) +
                self.b)

        h = tf.nn.dropout(activation_functions[activation](h_in),
                          self.dropout_keep_prob)

        p = tf.matmul(h, self.W2)

        cross_entropy = tf.divide(
                            tf.reduce_sum(
                                tf.nn.softmax_cross_entropy_with_logits(
                                    labels=trans_embed, logits=p)),
                            self.batch_size)

        regularized_params = list(map(tf.nn.l2_loss,
                                      [self.W_w, self.W_t, self.W_l, self.W2,
                                       self.b, self.word_embedding,
                                       self.tag_embedding,
                                       self.label_embedding]))

        l2_loss = l2_weight * tf.add_n(regularized_params)

        cost = tf.add(cross_entropy, l2_loss)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

        self.predicted_transitions = tf.nn.top_k(p, k=num_transitions)

        predicted_correctly = tf.equal(tf.argmax(p, 1), tf.argmax(trans_embed, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))

    def _to_batches(self, examples):
        all_input_words = []
        all_input_tags = []
        all_input_labels = []
        all_expected_outputs = []

        random.shuffle(examples)

        for relevant_ids, action in examples:
            all_input_words.append(relevant_ids["word"])
            all_input_tags.append(relevant_ids["tag"])
            all_input_labels.append(relevant_ids["label"])

            gold_transition = self.transition_vector.index_of(action)

            all_expected_outputs.append(gold_transition)

        def f():
            i = random.randrange(0, max(len(examples) - self.batch_size, 1))

            return (all_input_words[i:i + self.batch_size],
                    all_input_tags[i:i + self.batch_size],
                    all_input_labels[i:i + self.batch_size],
                    all_expected_outputs[i:i + self.batch_size])

        return f

    def train(self, trees, session, epochs=1000, dropout_keep_prob=0.5):
        examples = []

        for i, tree_ in enumerate(trees):
            if i % 500 == 0:
                print("Tree", i, file=sys.stderr)

            examples.extend(list(Oracle(tree_, self)))

        batcher = self._to_batches(examples)

        for epoch in range(epochs):
            batch = batcher()

            feed_dict = {
                self.input_words: batch[0],
                self.input_tags: batch[1],
                self.input_labels: batch[2],
                self.expected: batch[3],
                self.dropout_keep_prob: dropout_keep_prob,
            }

            session.run(self.optimizer, feed_dict=feed_dict)

            if epoch % 100 == 0:
                train_accuracy = session.run(self.accuracy, feed_dict=feed_dict)
                print('epoch {0}, training accuracy {1}'.format(epoch, train_accuracy))

    def parse(self, sentences, output, session):
        all_arcs = []
        for i, sentence in enumerate(sentences):
            print("Parsing", i, len(sentences))
            all_arcs.append(self.parse_sentence(sentence, session))

        with open(output, 'w') as f:
            for i, sent_arcs in enumerate(all_arcs):
                for h, m, rel in sorted(sent_arcs, key=lambda a: a.modifier):
                    if m.name != '*root*':
                        f.write("\t".join([str(m.index), m.name, '_', m.tag.name,
                                           m.tag.name, '_', str(h.index),
                                           rel.name,
                                           '_', '_']))
                        f.write('\n')
                f.write('\n')

    def parse_sentence(self, sentence, session):
        config = Configuration(self, sentence)

        while not config.finished:
            relevant_ids = config.relevant_ids

            feed_dict = {
                self.input_words: [relevant_ids["word"]],
                self.input_tags: [relevant_ids["tag"]],
                self.input_labels: [relevant_ids["label"]],
                self.dropout_keep_prob: 1.0,
            }

            transitions = session.run(self.predicted_transitions,
                                      feed_dict=feed_dict)

            named_transitions = []

            for transition in transitions.indices.squeeze(0):
                action = self.transition_vector[transition]

                named_transitions.append(action)

            feasible_transitions = filter(config.can_apply, named_transitions)

            # NOTE: this will raise an exception if feasible_transitions is
            # empty, but since that happening means there is a flaw in the
            # algorithm, we do want the program to crash and burn in that case
            best_action, best_label = next(feasible_transitions)

            if best_action == "shift":
                config.shift()
            elif best_action == "right":
                config.right_arc(best_label)
            else:
                config.left_arc(best_label)

        return config.tree.arcs
