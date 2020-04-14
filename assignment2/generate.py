import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

# NOTE ==============================================
# This is where your models are imported
# from models import RNN, GRU
# from models import make_model as TRANSFORMER

from solution_categorical import RNN, GRU
from solution_categorical import make_model as TRANSFORMER

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
# #############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=911,
                    help='random seed')

def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]
experiment_path = os.path.join(args.save_dir, '_'.join([argsdict['model'],
                                         argsdict['optimizer']]
                                         + flags))
# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.makedirs(experiment_path, exist_ok=True)

print ("\nPutting log in %s"%experiment_path)
argsdict['save_dir'] = experiment_path
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key+'    '+str(argsdict[key])+'\n')

# Set the random seed manually for reproducibility.
set_seed(args.seed)


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# LOADING & PROCESSING
#
# ##############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from '+args.data)
if args.data == 'SLURM_TMPDIR':
    raw_data = ptb_raw_data(data_path=os.environ['SLURM_TMPDIR'])
else:
    raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# MODEL SETUP
#
# ##############################################################################

# NOTE ==============================================
# This is where your model code will be called.
if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
    save_dir = '/home/soumye/Desktop/courses/ift6135/IFT6135H20_assignment/assignment2/runs/rnn.pt'
    # save_dir = '/home/makeshnarsimhan/IFT6135/updated_model_code_results/results/RNN/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_dir_resultsRNN_save_best_0/best_params.pt'

elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
    save_dir = '/home/soumye/Desktop/courses/ift6135/IFT6135H20_assignment/assignment2/runs/gru.pt'
    
    # save_dir = '/home/makeshnarsimhan/IFT6135/updated_model_code_results/results/GRU/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_dir_resultsGRU_save_best_0/best_params.pt'

elif args.model == 'TRANSFORMER':
    if args.debug:  # use a very small model
        model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
    else:
        # Note that we're using num_layers and hidden_size to mean slightly
        # different things here than in the RNNs.
        # Also, the Transformer also has other hyperparameters
        # (such as the number of attention heads) which can change it's behavior.
        model = TRANSFORMER(vocab_size=vocab_size, n_units=args.hidden_size,
                            n_blocks=args.num_layers, dropout=1.-args.dp_keep_prob)
    # these 3 attributes don't affect the Transformer's computations;
    # they are only used in run_epoch
    model.batch_size=args.batch_size
    model.seq_len=args.seq_len
    model.vocab_size=vocab_size
else:
    print("Model type not recognized.")
def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

print(save_dir)
model.load_state_dict(torch.load(save_dir, map_location=device))
model = model.to(device)
words = []
words.append(word_to_id['i'])
words.append(word_to_id['answer'])
words.append(word_to_id['universe'])
words.append(word_to_id['india'])
words.append(word_to_id['dog'])
words.append(word_to_id['forest'])
words.append(word_to_id['run'])
words.append(word_to_id['long'])
words.append(word_to_id['murder'])
words.append(word_to_id['old'])
print(word_to_id['<unk>'])
	

given = np.array(words,dtype=np.int64)
#given = np.array(list(id_2_word.keys())[20:30], dtype=np.int64)
#print(word_to_id.keys())
print([id_2_word[id] for id in given])
print("Given")
print(given)
given = torch.from_numpy(given).to(device)

hidden = model.init_hidden().to(device)
hidden = repackage_hidden(hidden)
model.eval()

#gen
#given = torch.tensor([1,2,3,4,5]).cuda()
res = model.generate(given, hidden, args.seq_len).transpose(0,1)
print(res.shape)
res = res.to('cpu').numpy()

res = [' '.join([id_2_word[id] for id in sent]) for sent in res]

with open('{}_{}_categorical.txt'.format(args.model, args.seq_len), 'w') as writer:
    for i, sent in enumerate(res):
        writer.write('{}. {}\n\n'.format(i+1, sent.replace('<unk>', '!unk!').replace('<eos>', '!eos!').replace('$', '\$')))
print('\n\n'.join(res))