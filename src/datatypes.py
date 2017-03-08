import warnings
from collections import namedtuple, defaultdict


Pair = namedtuple("Pair", ["id", "name"])


Arc = namedtuple("Arc", ["head", "modifier", "label"])


class Label(Pair):
    pass


class POSTag(Pair):
    pass


class Tree:

    def __init__(self, sentence, arcs=None, gold=False):
        self.sentence = sentence
        self.gold = gold

        self.arcs = []
        self._arcs_by_head = defaultdict(list)
        self._arcs_by_mod = {}

        if arcs is not None:
            self.arcs = arcs

            for arc in arcs:
                self._arcs_by_head[arc.head.index].append(arc)
                self._arcs_by_mod[arc.modifier.index] = arc

    def head_of(self, word):
        if isinstance(word, int):
            word_idx = word
        else:
            word_idx = word.index

        try:
            return self._arcs_by_mod[word_idx].head
        except IndexError:
            return None

    def modifiers(self, word):
        if isinstance(word, int):
            word_idx = word
        else:
            word_idx = word.index

        return sorted([arc.modifier for arc in self._arcs_by_head[word_idx]])

    def label_of(self, word):
        if isinstance(word, int):
            word_idx = word
        else:
            word_idx = word.index

        try:
            return self._arcs_by_mod[word_idx].label
        except KeyError:
            return None

    def add_arc(self, head, modifier, label):
        assert label is not None

        if self.gold:
            warnings.warn("You're modifying a gold parse tree")

        arc = Arc(head, modifier, label)

        self._arcs_by_head[arc.head.index].append(arc)
        self._arcs_by_mod[arc.modifier.index] = arc

        self.arcs.append(arc)

    def __repr__(self):
        return repr(self.arcs)


class Word:

    def __init__(self, id, name, index, tag):
        self.id = id
        self.name = name
        self.index = index

        self.tag = tag

    @classmethod
    def as_dummy(cls):
        return cls(-1, "*INEXISTENT*", -1, None)

    def __eq__(self, other):
        return self.index == other.index

    def __ne__(self, other):
        return self.index != other.index

    def __lt__(self, other):
        return self.index < other.index

    def __le__(self, other):
        return self.index <= other.index

    def __gt__(self, other):
        return self.index > other.index

    def __ge__(self, other):
        return self.index >= other.index

    def __repr__(self):
        return "Word({}, {}, {}, {})".format(self.id,
                                             self.name,
                                             self.index,
                                             self.tag)


class Sentence:

    def __init__(self, words):
        self.words = words
        self.pos_tags = [w.tag for w in words]

    def __repr__(self):
        return repr(self.words)


class Configuration:

    def __init__(self, model, sentence, gold_tree=None):
        self.gold_tree = gold_tree
        self.stack = []
        self.buffer = []
        self.tree = Tree(sentence)
        self.sentence = sentence
        self.buffer = list(reversed(self.sentence.words))

    @property
    def relevant_ids(self):
        word_feats = defaultdict(int)
        tag_feats = defaultdict(int)
        label_feats = defaultdict(int)

        for i in [1, 2, 3]:
            try:
                word_feats["s_%d" % i] = self.stack[-i].id
                tag_feats["s_%d" % i] = self.stack[-i].tag.id
            except IndexError:
                pass

        for i in [1, 2, 3]:
            try:
                word_feats["b_%d" % i] = self.buffer[-i].id
                tag_feats["b_%d" % i] = self.buffer[-i].tag.id
            except IndexError:
                pass

        for i in [1, 2]:
            try:
                word = self.stack[-i]
            except IndexError:
                break
            else:
                # the first and second letmost / rightmost children of word
                children = self.tree.modifiers(word)

                left = list(filter(lambda x: x.index < word.index, children))[:2]
                right = list(filter(lambda x: x.index > word.index, children))[-2:]

                left = left + [None] * (2 - len(left))
                right = [None] * (2 - len(right)) + right

                for j, w in enumerate(left):
                    key = "lc_%d(s_%d)" % (j+1, i)
                    if w is not None:
                        word_feats[key] = w.id
                        tag_feats[key] = w.tag.id
                        label_feats[key] = self.tree.label_of(w).id

        for i in [1, 2]:
            try:
                word = self.stack[-i]
            except IndexError:
                break
            else:
                children = self.tree.modifiers(word)

                left = list(filter(lambda x: x.index < word.index, children))
                right = list(filter(lambda x: x.index > word.index, children))

                for side, idx in [(left, 0), (right, -1)]:
                    try:
                        # this is lc_1(s_i) and rc_1(s_i)
                        farthest_child = side[idx]

                        grandchildren = self.tree.modifiers(farthest_child.index)

                        try:
                            # try to get lc_1(lc_1(s_i)) and rc_1(rc_1(s_i))
                            w = grandchildren[idx]

                            # note that just because we're looking at the
                            # leftmost element of grandchildren, that doesn't
                            # mean it's the leftmost child of farthest_child
                            # because farthest_child could have only 1 child and
                            # it's on the right side
                            if side is left and w.index < farthest_child.index:
                                key = "lc_1(lc_1(s_%d))" % i
                                word_feats[key] = w.id
                                tag_feats[key] = w.tag.id
                                label_feats[key] = self.tree.label_of(w).id

                            elif side is right and w.index > farthest_child.index:
                                key = "rc_1(rc_1(s_%d))" % i
                                word_feats[key] = w.id
                                tag_feats[key] = w.tag.id
                                label_feats[key] = self.tree.label_of(w).id

                        except IndexError:
                            pass

                    except IndexError:
                        pass

        word_keys = ["s_1", "s_2", "s_3", "b_1", "b_2", "b_3",
                     "lc_1(s_1)", "rc_1(s_1)", "lc_2(s_1)", "rc_2(s_1)",
                     "lc_1(s_2)", "rc_1(s_2)", "lc_2(s_2)", "rc_2(s_2)",
                     "lc_1(lc_1(s_1))", "rc_1(rc_1(s_1))",
                     "lc_1(lc_1(s_2))", "rc_1(rc_1(s_2))"]
        tag_keys = word_keys
        label_keys = word_keys[6:]

        words = [word_feats[key] for key in word_keys]
        tags = [tag_feats[key] for key in tag_keys]
        labels = [label_feats[key] for key in label_keys]

        return {
            "word": words,
            "tag": tags,
            "label": labels,
        }

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.tree.add_arc(self.stack[-1], self.stack[-2], label)
        del self.stack[-2]

    def right_arc(self, label):
        self.tree.add_arc(self.stack[-2], self.stack[-1], label)
        del self.stack[-1]

    def is_missing_child(self, word):
        """
        Return True if there is a child of word_idx in the gold tree which
        is missing from the current tree.
        """

        children = [w.index for w in self.tree.modifiers(word)]
        all_gold_children = [w.index for w in self.gold_tree.modifiers(word)]

        return bool(set(all_gold_children) - set(children))

    @property
    def finished(self):
        return (len(self.buffer) == 0 and
                len(self.stack) == 1 and
                self.stack[0].index == 0)

    def can_apply(self, action):
        name, label = action

        # NOTE: It might seem like a good idea to check if the label isn't *PAD*
        # but for whatever reason the UAS and LAS go down if you do.

        if name in ["right", "left"]:
            if name == "left":
                stack_pos = -1
            else:
                stack_pos = -2

            try:
                head = self.stack[stack_pos]
            except IndexError:
                head = Word.as_dummy()

            if head.index < 0:
                return False
            elif head.index == 0 and label.name != 'root':
                return False

        if name == "left":
            return len(self.stack) > 2
        elif name == "right":
            return ((len(self.stack) > 2) or
                    (len(self.stack) == 2 and len(self.buffer) == 0))
        elif name == "shift":
            return len(self.buffer) > 0
        else:
            # name might be None
            return False


# TESTED and it's fully functional
class TransitionVector:

    def __init__(self, relations):
        self.relations = relations
        self.labels = {i: Label(i, rel) for rel, i in self.relations.items()}

        self.actions = []
        for i in range(2*len(relations)):
            if i == 0:
                item = (None, Label(0, "*PAD*"))
            elif i < len(self.relations):
                item = ("left", self.labels[i])
            elif i < 2 * len(self.relations) - 1:
                item = ("right", self.labels[i - len(self.relations) + 1])
            else:
                item = ("shift", None)

            self.actions.append(item)

    def index_of(self, action):
        """
        Returns an index that can be used to create a onehot vector.
        """

        transition, label = action

        if transition == "shift":
            gold_transition = 2 * len(self.relations) - 1
        elif transition == "left":
            gold_transition = label.id
        elif transition == "right":
            gold_transition = label.id + len(self.relations) - 1
        elif transition is None and label.name == "*PAD*":
            gold_transition = 0
        else:
            raise Exception("Invalid transition")

        return gold_transition

    def __getitem__(self, idx):
        """
        Get the action for the given index.
        Returns (direction, label), where direction is None for the padding
        """

        return self.actions[idx]

    def __len__(self):
        return 2*len(self.relations)
