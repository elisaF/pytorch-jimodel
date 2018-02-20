from queue import Queue
from collections import defaultdict
import logging
import pickle


class Edu(object):
    def __init__(self):
        self.indices = []

    def __repr__(self):
        return '{0.indices}'.format(self)


class Doc(object):
    def __init__(self):
        self.edus = []
        self.order = []
        self.tree = defaultdict(list)  # map parent node to list of children nodes
        self.relations = {}  # map edu index to relation index
        self.root = None  # root node
        self.label = -1  # document label
        self.filename = None

    def __repr__(self):
        template = '\nEDUs: {0.edus} ' \
                   '\nOrder: {0.order}' \
                   '\nParent to child mapping: {0.tree}' \
                   '\nRelations: {0.relations}' \
                   '\nRoot: {0.root}' \
                   '\nLabel: {0.label}' \
                   '\nFilename: {0.filename}'
        return template.format(self)


class Corpus(object):
    def __init__(self):
        self.docs = []

    def size(self):
        return len(self.docs)


def read_corpus(filename, dictionary, b_update):
    logging.info("Reading data from: %s", filename)
    corpus = Corpus()
    doc = Doc()
    with open(filename) as f:
        next(f)  # skip header
        for line in f:
            if line[0] != '=':
                #  within document
                items = line.split("\t")
                eidx = int(items[0])
                pidx = int(items[1])
                ridx = int(items[2])
                edu = read_edu(items[3], dictionary, b_update)
                doc.edus.append(edu)  # store the edu
                doc.tree[pidx].append(eidx)  # store the pnode index
                doc.relations[eidx] = ridx  # relation index
                if pidx == -1:
                    doc.root = eidx  # root node
            else:
                # end of document
                items = line.split("\t")
                doc.filename = items[1]  # get filename
                doc.label = int(items[2])  # get label
                if len(doc.edus) > 0:
                    # before saving this doc, get the topological order
                    doc.order = topological_sorting(doc)
                    # save this doc
                    corpus.docs.append(doc)
                else:
                    logging.warning("Empty doc: %s", doc.filename)
                doc = Doc()  # reset this variable

        if len(doc.edus) > 0:
            doc.order = topological_sorting(doc)
            print_int_vector(doc.order)
            corpus.docs.append(doc)

        logging.info("Read %s docs; the vocab has %s types.", len(corpus.docs), len(dictionary.vocab_dict.keys()))
        return corpus


def read_edu(edu_text, dictionary, b_update):
    tokens = edu_text.split(" ")
    edu = Edu()
    for tok in tokens:
        if b_update or tok in dictionary.vocab_dict:
            edu.indices.append(dictionary.convert(tok))
        else:
            edu.indices.append(dictionary.convert("UNK"))
    if len(edu.indices) == 0:
        # just in case there is a weird empty sentence
        edu.indices.append(dictionary.convert("UNK"))
    return edu


def topological_sorting(doc):
    pnode_list = []
    q = Queue(maxsize=0)
    q.put(doc.tree[-1][0])  # add the root node
    while not q.empty():
        pidx = q.get()
        pnode_list.append(pidx)
        for v in doc.tree[pidx]:
            q.put(v)
    pnode_list.reverse()
    return pnode_list


def print_int_vector(vec):
    for val in vec:
        print(val, " ")
    print("\n")


class Dictionary:
    def __init__(self, frozen=False, map_unk=False, unk_id=-1):
        self.frozen = frozen
        self.map_unk = map_unk
        self.unk_id = unk_id
        self.vocab_dict = {}
        self.words = []

    def convert(self, word):
        if word not in self.vocab_dict:
            if self.frozen:
                if self.map_unk:
                    return self.unk_id
                else:
                    logging.error("Unknown word encountered in frozen dictionary: %s", word)
            self.words.append(word)
            self.vocab_dict[word] = len(self.words) - 1
        return self.vocab_dict[word]

    def freeze(self):
        self.frozen = True

    def size(self):
        return len(self.words)

    def load_dict(self, fname):
        self.vocab_dict = pickle.load(open(fname, 'rb'))
        self.words = self.vocab_dict.keys()

    def save_dict(self, fname):
        pickle.dump(self.vocab_dict, open(fname, 'wb'))

