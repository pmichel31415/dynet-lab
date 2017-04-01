from __future__ import print_function, division
import numpy as np
import numpy.random as npr
import dynet as dy
import time
import sys
import os.path
sys.path.append(os.path.abspath(os.path.realpath(__file__) + '/../../util/'))
import options
import data

# Retrieve command line options
opt = options.mlp_parser().parse_args()
# 
if opt.model_out is None:
    if opt.model_in is None:
        opt.model_out = opt.exp_name + '_model.txt'
    else:
        opt.model_out = opt.model_in
else:
    model_out = opt.model_out

# print config
if opt.verbose:
    options.print_config(opt)

# Load data
N_train = 55000
N_dev = 5000
N_test = 10000
train_x, dev_x, test_x, train_y, dev_y, test_y = data.get_mnist(N_dev, shuffle=opt.shuffle, preprocessing=data.whiten)

# Model parameters
num_classes = len(set(train_y)) # Number of classes
input_dim = train_x.shape[1]    # Dimension of the input
dims = [input_dim] + [int(dh) for dh in opt.hidden_dim.split('-')]+[num_classes]
num_layers = len(dims) - 1      # Number of layers

# Configure the activation functions
if opt.activation == 'sigmoid': # Sigmoid
    activation = dy.logistic    # Dynet operation
    gain = 1/4                  # Gain used for the initialization (f'(0))
elif opt.activation == 'tanh':  # Tanh
    activation = dy.tanh        # Dynet operation
    gain = 1                    # Gain used for the initialization (f'(0))
elif opt.activation == 'relu':  # ReLU
    activation = dy.rectify     # Dynet operation
    gain = 1 / 2                # Se He et al. (2015)

# Create model
model = dy.Model()                      # DyNet Model
trainer = dy.SimpleSGDTrainer(model,    # Trainer
                              opt.learning_rate,
                              opt.learning_rate_decay)
trainer.set_clip_threshold(-1)          # Disable gradient clipping

# Create the parameters
params = []                                                                 # This holds the parameters for each layer
for i, (di, do) in enumerate(zip(dims, dims[1:])):                          # Iterate over the input/output dimensions for each layer
    var = 2 / (di+do) / (1 if (i == num_layers - 1) else gain)              # Variance for the initialization (See Glorot, Bengio (2011))
    W_p = model.add_parameters((do, di), init=dy.NormalInitializer(0, var)) # Sample weights
    b_p = model.add_parameters((do,), init=dy.ConstInitializer(0))          # Initialize biases at 0
    params.append((W_p, b_p))                                               # Add to the list

# Load existing model
if opt.model_in is not None:
    print('Loading from file:', opt.model_in)
    params_list = model.load(opt.model_in)
    params = [(W_p, b_p) for W_p, b_p in zip(params_list[:num_layers], params_list[num_layers:])]


def run_MLP(x):
    """
    Runs MLP to get the last layer before softmax
    """
    bsize, d = x.shape
    h = dy.inputTensor(x.T, batched=True)               # Initialize layer value
    for i, (W_p, b_p) in enumerate(params):             # Iterate over layers
        W, b = W_p.expr(), b_p.expr()                   # Load parameters in computation graph
        a = W * h + b                                   # Affine transform
        h = a if (i == num_layers-1) else activation(a) # Apply non-linearity (except for last layer)
    return h


def get_loss(x, y):
    """
    Get loss -log(softmax(score[y]))
    """
    score = run_MLP(x)
    bsize = x.shape[0]
    return dy.sum_batches(dy.pickneglogsoftmax_batch(score, y)) / bsize


def get_probabilities(x, stats=False, noise=0.0):
    """
    Get probabilities for each class
    """
    score = run_MLP(x)
    p = dy.softmax(score)
    return p.npvalue()


def get_accuracy(x, y, stats=False, noise=0.0):
    """
    Get prediciton accuracy on batch
    """
    prob = get_probabilities(x)
    acc = (np.argmax(prob, axis=0) == y).sum() / len(y)
    return acc


def get_trainbatch():
    """
    Sample a training minibatch at random
    """
    idx = npr.choice(N_train, size=opt.batch_size)
    return train_x[idx], train_y[idx]


def train_batch():
    """
    Perform training on a minibatch
    """
    start = time.time()                     # Time the iteration
    x_batch, y_batch = get_trainbatch()     # Get training batch
    dy.renew_cg()                           # Renew Dynet computation graph
    loss_expr = get_loss(x_batch, y_batch)  # Get loss expression
    loss_value = loss_expr.value()          # Run forward pass
    loss_expr.backward()                    # Run backward pass
    trainer.update()                        # Update parameters
    if opt.verbose:
        print('Iteration:', iteration,
              ', elapsed: %.2f s' % (time.time() - start),
              'training loss:', (loss_value))


def validate():
    """
    Evaluate the model on the validation set
    """
    start = time.time()                         # Time the iteration
    accuracy = 0
    global best_dev_acc
    for i in range(0, N_dev, opt.batch_size):
        idx = range(i, i+opt.batch_size)        # Get minibatch
        x_batch, y_batch = dev_x[idx], dev_y[idx]
        dy.renew_cg()                           # Renew Dynet computation graph
        acc = get_accuracy(x_batch, y_batch)    # Compute accuracy
        accuracy += acc                         # Add to the overall accuracy
    accuracy /= (N_dev/opt.batch_size)          # Average the accuracy
    trainer.status()                            # Print trainer status
    print('Iteration:', iteration,
          ', elapsed: %.2f s' % (time.time() - start),
          'dev accuracy: %.2f' % (100 * accuracy))
    if best_dev_acc < accuracy:                 # Save the model if the accuracy is maximal
        best_dev_acc = accuracy
        print('Saving model with dev accuracy: %.2f' %
              (best_dev_acc * 100), 'to file', opt.model_out)
        model.save(opt.model_out, [W_p for W_p, _ in params] + [b_p for _, b_p in params])
    trainer.update_epoch()                      # Update learning rate


def test():
    """
    Evaluate the model on the test data
    """
    start = time.time()                         # Time the iteration
    accuracy = 0
    for i in range(0, N_test, opt.batch_size):
        idx = range(i, i + opt.batch_size)      # Get minibatch
        x_batch, y_batch = test_x[idx], test_y[idx]
        dy.renew_cg()                           # Renew Dynet computation graph
        acc = get_accuracy(x_batch, y_batch)    # Compute accuracy on minibatch
        accuracy += acc                         # Add to the overall accuracy
    accuracy /= (N_test/opt.batch_size)         # Average the accuracy
    print('Test at iteration:', iteration,      # Print the results
          ', elapsed: %.2f s' % (time.time() - start),
          'test accuracy: %.2f' % (100 * accuracy))

if __name__ == '__main__':  # Training
    iteration = 0
    best_dev_acc = 0
    for iteration in range(1, opt.num_iters + 1):
        train_batch()       # Train on training batch
        if iteration % opt.valid_every == 0:
            validate()      # Test on validation data
        if iteration % opt.test_every == 0:
            test()          # Test on test data
