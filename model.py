import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import random, time
torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, nclasses, embedding_dim, hidden_dim, droprate, batch_size=1):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(droprate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM to the label
        self.hidden2label = nn.Linear(hidden_dim, self.nclasses)

    def init_hidden(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are num of layers * num directions x batch size x hidden dim
        #if self.use_gpu:
        #    h0 = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        #    c0 = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        #else:
        h0 = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2))
        c0 = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def _get_edu_reps(self, doc):
        self.hidden = self.init_hidden()
        # get embeddings
        # batch size should be number of EDUs in doc, seq_len should be max length of edu in current doc
        edu_reps = autograd.Variable(torch.FloatTensor(len(doc.edus), self.batch_size, self.hidden_dim).zero_())
        for idx, edu in enumerate(doc.edus):
            # print("Edu: ", len(edu.indeces), edu.indeces)
            embeds = self.word_embeds(autograd.Variable(torch.LongTensor(edu.indeces)))
            # print("Word embeddings: ", embeds.size())
            embeds_dropout = self.dropout(embeds)
            # print("After dropout: ", embeds_dropout.size())
            # get output for each timestep in sequence
            # sequence len x batch x hidden size*num directions
            # ? x 1 x ?*2
            # reshape input embeddings to (seq_len x batch x input embedding dim) <- max seq_len
            embeds_dropout_reshaped = embeds_dropout.view(len(embeds_dropout), self.batch_size, -1)
            # print("After reshape: ", embeds_dropout_reshaped.size())
            lstm_out, self.hidden = self.lstm(embeds_dropout_reshaped, self.hidden)
            # get last hidden state from fwd and bkwd LSTM
            last_fwd = self.hidden[0][0]
            last_bwd = self.hidden[0][1]
            edu_rep = torch.cat((last_fwd, last_bwd), dim=1)
            # print("LSTM last hidden: ", edu_rep.size())  # 1x32
            edu_reps[idx] = edu_rep
        #print("Returning all edu reps: ", edu_reps.size())
        return edu_reps

    def forward(self, doc):
        edu_reps = self._get_edu_reps(doc)
        edu_average = torch.mean(edu_reps, dim=0)
        #print("Average: ", edu_average.size(), edu_average.grad_fn)
        y = self.hidden2label(edu_average)
        return y


def train(trncorpus, vocab_size, nclasses, embedding_dim, hidden_dim, droprate, niter):
    model = BiLSTM(vocab_size, nclasses, embedding_dim, hidden_dim, droprate)
    softmax_function = nn.LogSoftmax()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    print("Start training")
    report_freq = 445
    order = list(range(trncorpus.size()))
    report = 0
    sample_counter = trncorpus.size()
    niter = niter * trncorpus.size() / report_freq
    while report < niter:
        start_time = time.time()
        complete_loss = 0
        for i in range(report_freq):
            # shuffle only on first pass and once we've gone through the whole corpus
            if sample_counter == trncorpus.size():
                sample_counter = 0
                print("*** SHUFFLE ***")
                random.shuffle(order)

            # build graph for this instance
            doc = trncorpus.docs[order[sample_counter]]
            sample_counter += 1
            # print("Training on doc ", repr(doc))
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            target = autograd.Variable(torch.LongTensor([doc.label]))

            # Step 3. Run our forward pass.
            pred_target = model(doc)
            # print("Pred target: ", pred_target)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(softmax_function(pred_target), target)
            complete_loss += loss.data
            loss.backward()
            optimizer.step()
            end_time = time.time()
            # print("Loss: ", loss.data)
        report += 1
        print("Total loss: ", complete_loss)
        print("Finished report ", report, "(", report / float(niter), ") in ", end_time-start_time)
        # Check predictions after training
        num_correct = 0
        preds = []
        labels = []
        for doc in trncorpus.docs:
            output = model(doc)
            # calc acc
            _, predicted = torch.max(output.data, 1)
            if doc.label == predicted[0]:
                num_correct += 1
            preds.append(predicted[0])
            labels.append(doc.label)
        print("Preds vs .labels:\n", preds[:50], "\n", labels[:50])
        print("Accuracy on training: ", num_correct / trncorpus.size())
    print("Done training")
    return model


def evaluate(model, evalcorpus):
    # Check predictions after training
    for idx, doc in enumerate(evalcorpus.docs):
        output = model(doc)
        # calc acc
        _, predicted = torch.max(output.data, 1)
        print("Predicted[", idx, "]: ", predicted)


def save_model(model, fname):
    torch.save(model.state_dict(), fname)


def load_model(fname):
    model = BiLSTM(*args, **kwargs)
    model.load_state_dict(torch.load(fname))
    return model
