import argparse
import os
import logging
import model
import util
import torch

def main():
    ###############################################################################
    # Load data
    ###############################################################################
    d = util.Dictionary()
    if args.task == "train":
        trncorpus = util.read_corpus(args.ftrn, d, True)
        d.freeze()  # no new word types allowed
        vocab_size = d.size()
        print("Train docs: ", trncorpus.size())
        logging.info("[TextClass] Train vocab size: %s", vocab_size)
        # save dict
        d.save_dict(fprefix+".dict")
        # read dev corpus
        devcorpus = util.read_corpus(args.fdev, d, False)
    elif args.task == "test":
        d.load_dict(args.fdct)
        d.freeze()
        vocab_size = d.size()
        logging.info("[TextClass] Test vocab size: %s", vocab_size)
        # load test corpus
        tstcorpus = util.read_corpus(args.ftst, d, False)

    ###############################################################################
    # Build the model
    ###############################################################################
    #if args.fmod:
    #    # load pre-trained model
    #    with open(args.fmod, 'rb') as f:
    #        model = torch.load(f)

    trained = model.train(trncorpus, vocab_size, args.nclass, args.inputdim, args.hiddendim, args.droprate, args.niter)
    #tc = textclass.TextClass(trncorpus, devcorpus, tstcorpus,
    #                         args.inputdim, args.hiddendim, args.nlayer, args.nclass, args.ndisrela, vocab_size, d,
    #                         args.arch, args.trainer, args.lr, args.droprate, args.evalfreq, args.niter, args.verbose,
    #                         args.fembed, model)
    model.evaluate(trained, devcorpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="task (for train, also specify training, dev files; "
                                       "for test. also specify dict, test, model files)",
                        choices=['train', 'test'], default='train')
    parser.add_argument("--ftrn", help="training file")
    parser.add_argument("--fdev", help="dev file")
    parser.add_argument("--ftst", help="test file")
    parser.add_argument("--fdct", help="dict file")
    parser.add_argument("--fmod", help="model file")
    parser.add_argument("--arch", help="model architecture", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--nclass", help="number of doc classes", type=int, default=5)
    parser.add_argument("--ndisrela", help="number of discourse relations", type=int, default=43)
    parser.add_argument("--inputdim", help="input dimension", type=int, default=64)
    parser.add_argument("--hiddendim", help="hidden dimension", type=int, default=64)
    parser.add_argument("--nlayer", help="number of hidden layers", type=int, default=1)
    parser.add_argument("--trainer", help="training method", choices=['sgd', 'adagrad', 'adam'], default='adagrad')
    parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
    parser.add_argument("--droprate", help="dropout rate", type=float, default=0.0)
    parser.add_argument("--niter", help="number of passes on the training set", type=int, default=1)
    parser.add_argument("--evalfreq", help="evaluation frequency on dev data", type=int, default=1)
    parser.add_argument("--fembed", help="word embedding file")
    parser.add_argument("--path", help="path to save files", default=".")
    parser.add_argument("-v", "--verbose", help="print training information", action="store_true")
    args = parser.parse_args()

    # check arguments
    if args.task == "train" and ((not args.ftrn) or (not args.fdev)):
        raise Exception("For training, you must specify both training file and dev file!")
    elif args.task == "test" and ((not args.ftst) or (not args.fdct) or (not args.fmod)):
        raise Exception("For testing, you must specify test file, dict file and model file!")

    # create file name for logging
    fprefix = os.path.join(args.path, "record-pid" + repr(os.getpid()))
    flog = fprefix + ".log"
    logging.basicConfig(filename=flog, level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # create output dir if needed
    if not os.path.exists(args.path):
        os.makedirs(args.path)
        logging.info("Successfully created folder: ", args.path)

    logging.info("[TextClass] training file: %s", args.ftrn)
    logging.info("[TextClass] dev file: %s", args.fdev)
    logging.info("[TextClass] test file: %s", args.ftst)
    logging.info("[TextClass] model file: %s", args.fmod)
    logging.info("[TextClass] model architecture: %s", args.arch)
    logging.info("[TextClass] number of doc classes: %s", args.nclass)
    logging.info("[TextClass] number of discourse relations: %s", args.ndisrela)
    logging.info("[TextClass] input dimension: %s", args.inputdim)
    logging.info("[TextClass] hidden dimension: %s", args.hiddendim)
    logging.info("[TextClass] number of hidden layers: %s", args.nlayer)
    logging.info("[TextClass] training method: %s", args.trainer)
    logging.info("[TextClass] learning rate: %s", args.lr)
    logging.info("[TextClass] number of iterations: %s", args.niter)
    logging.info("[TextClass] dropout rate (0: no dropout): %s", args. droprate)
    logging.info("[TextClass] evaluation frequency on dev data: %s", args.evalfreq)
    logging.info("[TextClass] word embedding file: %s", args.fembed)
    logging.info("[TextClass] output path: %s", args.path)
    logging.info("[TextClass] verbose: %s", args.verbose)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(1)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(1)

    main()

