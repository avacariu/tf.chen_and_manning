from datatypes import Configuration, Word


class Oracle:

    def __init__(self, tree, model):
        self.tree = tree
        self.model = model
        self.config = Configuration(self.model, self.tree.sentence, self.tree)

    def __iter__(self):
        return self

    def __next__(self):

        if not self.config.finished:
            try:
                w1 = self.config.stack[-2]
            except IndexError:
                w1 = Word.as_dummy()

            try:
                w2 = self.config.stack[-1]
            except IndexError:
                w2 = Word.as_dummy()

            relevant_ids = self.config.relevant_ids
            next_transition = None

            if w1.index > 0 and self.tree.head_of(w1) == w2:
                next_transition = ("left", self.tree.label_of(w1))
                self.config.left_arc(next_transition[1])

            elif (w1.index >= 0 and
                    self.tree.head_of(w2) == w1 and
                    not self.config.is_missing_child(w2)):

                next_transition = ("right", self.tree.label_of(w2))
                self.config.right_arc(next_transition[1])

            else:
                next_transition = ("shift", None)
                self.config.shift()

            return relevant_ids, next_transition
        else:
            raise StopIteration
