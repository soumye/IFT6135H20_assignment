import pickle
import numpy as np
import gzip
import time
from tqdm import tqdm

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]] 
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10) 

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]] 
    test_data = list(zip(test_inputs, test_data[1])) 
    return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()

class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 init_method='glorot',
                 activation="relu",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            if self.init_method == "zeros":
                self.weights[f"W{layer_n}"] = np.zeros((all_dims[layer_n - 1], all_dims[layer_n]))
            elif self.init_method == "normal":
                # Normal
                self.weights[f"W{layer_n}"] = np.random.normal(size=(all_dims[layer_n - 1], all_dims[layer_n]))
            elif self.init_method == "glorot":
                # Glorot
                d = np.sqrt(6.0/(all_dims[layer_n-1] + all_dims[layer_n]))
                print("Glorot", d)
                self.weights[f"W{layer_n}"] = np.random.uniform(low=-d, high=d,size=(all_dims[layer_n - 1], all_dims[layer_n]))
            else:
                raise Exception("invalid")

    def relu(self, x, grad=False):
        val = x*(x>0)
        if grad:
            # WRITE CODE HERE
            return 1.0*(x>0)
        # WRITE CODE HERE
        return val

    def sigmoid(self, x, grad=False):
        val = 1.0/(1.0+np.exp(-x))
        if grad:
            # WRITE CODE HERE
            return val*(1.0-val) 
        # WRITE CODE HERE
        return val

    def tanh(self, x, grad=False):
        exp = np.exp(2*x)
        val = (exp-1.0)/(exp+1.0)
        if grad:
            # WRITE CODE HERE
            return (1.0+val)(1.0-val)
        # WRITE CODE HERE
        return val

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        exps =  np.exp(x-np.max(x,1).reshape(-1,1))
        return exps/np.sum(exps,1).reshape(-1,1)

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        for layer_n in range(1, self.n_hidden + 2):
            cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"]) 
        cache[f"Z{self.n_hidden + 1}"] = self.softmax(cache[f"A{self.n_hidden + 1}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        
        # # start = time.process_time()
        grads[f"dA{self.n_hidden+1}"] = -(labels - cache[f"Z{self.n_hidden + 1}"])
        for layer_n in range(self.n_hidden+1, 0, -1):
            #grads[f"dW{layer_n}"] = np.matmul(grads[f"dA{layer_n}"],cache[f"Z{layer_n-1}"].T)
            
            # # print('20',time.process_time() - start)
            # import ipdb; ipdb.set_trace()
            # # start = time.process_time()
            grads[f"dW{layer_n}"] = grads[f"dA{layer_n}"][:,None,:] * cache[f"Z{layer_n-1}"][:,:,None]
            # # print('21',time.process_time() - start)
            
            # # start = time.process_time()
            grads[f"db{layer_n}"] = grads[f"dA{layer_n}"]
            # # print('22',time.process_time() - start)

            # # start = time.process_time()
            #grads[f"dZ{layer_n-1}"] =  np.einsum("ij,kj->ki",self.weights[f"W{layer_n}"],grads[f"dA{layer_n}"])
            grads[f"dZ{layer_n-1}"] = np.matmul(grads[f"dA{layer_n}"],self.weights[f"W{layer_n}"].T)
            # # print('23',time.process_time() - start)

            # # start = time.process_time()
            if layer_n > 1:
                grads[f"dA{layer_n-1}"] = grads[f"dZ{layer_n-1}"]*self.activation(cache[f"A{layer_n-1}"], grad=True)
            # # print('24',time.process_time() - start)
        return grads

    def update(self, grads):
        # # start = time.process_time()
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            # # print('1',time.process_time() - start)
            # # start = time.process_time()
            bsz = grads[f"dW{layer_n}"].shape[0]
            self.weights[f"W{layer_n}"] -= self.lr *np.sum(grads[f"dW{layer_n}"],0)/bsz
            # # print('2',time.process_time() - start)
            # # start = time.process_time()
            self.weights[f"b{layer_n}"] -= self.lr *np.sum(grads[f"db{layer_n}"],0)/bsz
            # # print('3',time.process_time() - start)
            

    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        return np.sum(-np.log(prediction[labels == 1.]))/prediction.shape[0]

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)
        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
        for epoch in range(n_epochs):
            for batch in tqdm(range(n_batches)):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                # # start = time.process_time()
                cache = self.forward(minibatchX)
                # # print("forward",time.process_time() - start)
                
                # # start = time.process_time()
                grads = self.backward(cache, minibatchY)
                # # print("backward",time.process_time() - start)

                # # start = time.process_time()
                self.update(grads)
                # # print("update grads", time.process_time() - start)
            
            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)
            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)
            print("Training Accuracy {}, Validation Accuracy {}, Training Loss {}, Validation Loss {}".format(train_accuracy, valid_accuracy, train_loss, valid_loss))

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy