import re

from collections import defaultdict, namedtuple
from datatypes import Tree, Word, Sentence, Label, POSTag, Arc


ConllEntry = namedtuple("ConllEntry", "id, word, pos, parent, relation")


def to_tree_and_sentence(conll_entries, vocab, tags, labels):

    words = [None] * len(conll_entries)

    for conll_entry in conll_entries:
        word = conll_entry.word
        tag = conll_entry.pos
        mod_id = conll_entry.id

        words[mod_id] = Word(vocab[word], word, mod_id, POSTag(tags[tag], tag))

    sentence = Sentence(words)
    arcs = []

    for conll_entry in conll_entries:
        modifier = words[conll_entry.id]
        head = words[conll_entry.parent]

        relation = conll_entry.relation
        label = Label(labels[relation], relation)

        arcs.append(Arc(head, modifier, label))

    tree = Tree(sentence, arcs, gold=True)

    return sentence, tree


def is_projective(sentence):
    roots = sentence[:]

    unassigned = defaultdict(int)
    for entry in sentence:
        for possible_child in sentence:
            if entry.id == possible_child.parent:
                unassigned[entry.id] += 1

    for _ in range(len(sentence)):
        for i in range(len(roots) - 1):
            if roots[i].parent == roots[i+1].id and unassigned[roots[i].id] == 0:
                unassigned[roots[i+1].id] -= 1
                del roots[i]
                break
            if roots[i+1].parent == roots[i].id and unassigned[roots[i+1].id] == 0:
                unassigned[roots[i].id] -= 1
                del roots[i+1]
                break

    return len(roots) == 1


def extract_vocab(conll_file):
    words = {'*root*'}
    pos = {'ROOT-POS'}
    rels = {'rroot'}

    with open(conll_file, 'r') as f:
        for line in f:
            entry = line.strip().split()
            if entry:
                words.add(normalize(entry[1]))
                pos.add(entry[3].upper())
                rels.add(entry[7])

    return words, pos, rels


def read_conll(fh, vocab, tags, relations, only_projective=False):
    sentences = []
    trees = []

    root = ConllEntry(0, '*root*', 'ROOT-POS', 0, 'rroot')
    tokens = [root]
    for line in fh:
        tok = line.strip().split()

        if tok == []:
            if len(tokens) > 1:
                if not only_projective or is_projective(tokens):
                    sentence, tree = to_tree_and_sentence(tokens, vocab, tags, relations)
                    sentences.append(sentence)
                    trees.append(tree)
            tokens = [root]

        else:
            tokens.append(ConllEntry(int(tok[0]),
                                     normalize(tok[1]),
                                     tok[3].upper(),
                                     int(tok[6]),
                                     tok[7]))
    if len(tokens) > 1:
        sentence, tree = to_tree_and_sentence(tokens, vocab, tags, relations)
        sentences.append(sentence)
        trees.append(tree)

    return sentences, trees


# The regex is a parameter so it is compiled at def execution time
def normalize(word, regex=re.compile(r"[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+")):
    if regex.match(word):
        return '*NUM*'

    return word.lower()


def return1():
    """A alternative for `lambda: 1` useful for pickling defaultdicts"""
    return 1
