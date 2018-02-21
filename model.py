import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import random, time
from cuda import ftype, itype

def prepare_sequence(edus):
    # sort, pad with zeros, then transpose
    edus.sort(key=len, reverse=True)
    edu_lengths = [len(edu) for edu in edus]
    seq_tensor = torch.zeros((len(edus), edu_lengths[0])).type(itype)
    for idx, (seq, seqlen) in enumerate(zip(edus, edu_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq).type(itype)
    return seq_tensor.transpose(0, 1), edu_lengths


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, nclasses, embedding_dim, hidden_dim, nlayers, droprate):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.vocab_size = vocab_size
        self.nclasses = nclasses
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(droprate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=self.nlayers, bidirectional=True)

        # Maps the output of the LSTM to the label
        self.hidden2label = nn.Linear(hidden_dim, self.nclasses)

    def init_hidden(self, batch_size):
        h0 = autograd.Variable(torch.zeros(self.nlayers*2, batch_size, self.hidden_dim // 2))
        c0 = autograd.Variable(torch.zeros(self.nlayers*2, batch_size, self.hidden_dim // 2))
        return (h0.cuda(), c0.cuda())

    def _get_edu_reps(self, doc):
        edus = [edu.indices for edu in doc.edus]
        padded_edus, edu_lengths = prepare_sequence(edus)
        self.hidden = self.init_hidden(padded_edus.size(-1))
        embeds = self.word_embeds(autograd.Variable(torch.LongTensor(padded_edus).type(itype)).cuda())
        embeds_dropout = self.dropout(embeds)
        embeds_packed = pack_padded_sequence(embeds_dropout, edu_lengths)
        lstm_out, (ht, ct) = self.lstm(embeds_packed, self.hidden)
        # concat last hidden state from fwd and bkwd LSTM
        edu_reps = torch.cat((ht[0], ht[1]), dim=1)
        return edu_reps

    def forward(self, doc):
        edu_reps = self._get_edu_reps(doc)
        edu_average = torch.mean(edu_reps, dim=0)
        y = self.hidden2label(edu_average.view(1, -1))
        return y


def train(trncorpus, devcorpus, vocab_size, nclasses, embedding_dim, hidden_dim, nlayers,
          trainer, lr, droprate, niter, report_freq, verbose):
    model = BiLSTM(vocab_size, nclasses, embedding_dim, hidden_dim, nlayers, droprate)
    model.cuda()
    softmax_function = nn.LogSoftmax()
    loss_function = nn.NLLLoss()
    if trainer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif trainer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif trainer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # create dev sample for reporting
    dev_sample = random.sample(devcorpus.docs, report_freq)

    logging.info("Start training")
    order = list(range(trncorpus.size()))
    report = 0
    epoch_counter = 0
    best_dev_accuracy = 0.0
    sample_counter = trncorpus.size()
    niter = niter * trncorpus.size() / report_freq
    while report < niter:
        start_time = time.time()
        complete_loss = 0.0
        for i in range(report_freq):
            # shuffle only on first pass and once we've gone through the whole corpus
            if sample_counter == trncorpus.size():
                sample_counter = 0
                epoch_counter += 1
                logging.info("*** Starting new epoch %s", epoch_counter)
                random.shuffle(order)

            # build graph for this instance
            doc = trncorpus.docs[order[sample_counter]]
            sample_counter += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            target = autograd.Variable(torch.LongTensor([doc.label]).type(itype)).cuda()

            # Step 3. Run our forward pass.
            pred_target = model(doc)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(softmax_function(pred_target), target)
            complete_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            end_time = time.time()
        report += 1
        if verbose:
            logging.info("Loss: %s", complete_loss)
            logging.info("Finished report %s (%s) in %s seconds.", report, report / float(niter), end_time-start_time)
            # Check predictions after training
            dev_accuracy = evaluate(model, dev_sample)
            logging.info("Accuracy on training: %s (%s)", dev_accuracy, best_dev_accuracy)
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
    logging.info("Done training")
    return model


def evaluate(model, docs):
    num_correct = 0
    preds = []
    labels = []
    for doc in docs:
        output = model(doc)
        _, predicted = torch.max(output.data, 1)
        if doc.label == predicted[0]:
            num_correct += 1
        preds.append(predicted[0])
        labels.append(doc.label)
    logging.info("Preds vs .labels:\n%s \n%s", preds[:50], labels[:50])
    return num_correct / len(docs)


def save_model(model, fname):
    torch.save(model.state_dict(), fname)


def load_model(fname, vocab_size, nclasses, embedding_dim, hidden_dim, nlayers, droprate):
    model = BiLSTM(vocab_size, nclasses, embedding_dim, hidden_dim, nlayers, droprate)
    model.cuda()
    model.load_state_dict(torch.load(fname))
    return model

