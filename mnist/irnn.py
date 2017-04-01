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
opt = options.rnn_parser().parse_args()
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
input_length = train_x.shape[1] # Dimension of the input
dh= opt.hidden_dim
di= 1

# Create model
model = dy.Model()                      # DyNet Model
trainer = dy.SimpleSGDTrainer(model,    # Trainer
                              opt.learning_rate,
                              opt.learning_rate_decay)
trainer.set_clip_threshold(-1)          # Disable gradient clipping

# Create the parameters
Wx_p = model.add_parameters((dh, di), init=dy.NormalInitializer(0, 0.001))                      # Sample weights
Wh_p = model.add_parameters((dh, dh), init=dy.IdentityInitializer())                            # Sample weights
bh_p = model.add_parameters((dh,), init=dy.ConstInitializer(0))                                 # Initialize biases at 0
A_p = model.add_parameters((num_classes, dh), init=dy.NormalInitializer(0, 1/(dh+num_classes))) # Sample weights
b_p = model.add_parameters((num_classes,), init=dy.ConstInitializer(0))                         # Initialize biases at 0

# Load existing model
if opt.model_in is not None:
    print('Loading from file:', opt.model_in)
    Wx_p, Wh_p, bh_p, A_p, b_p = model.load(opt.model_in)


def run_IRNN(x):
    """
    Runs MLP to get the last layer before softmax
    """
    bsize, d = x.shape
    Wh, Wx, bh = Wh_p.expr(), Wx_p.expr(), bh_p.expr()                   # Load parameters in computation graph
    A,b=A_p.expr(),b_p.expr()
    x_list = [dy.inputTensor(x_t, batched=True) for x_t in x.T]               # Initialize layer value
    h=dy.zeroes((dh,),batch_size=bsize)# Initialize layer value
    for x_t in x_list:             # Iterate over layers
        a = Wh * h + Wx * x_t + bh                                   # Affine transform
        h = dy.rectify(a) # Apply non-linearity (except for last layer)
    return A * h + b


def get_loss(x, y):
    """
    Get loss -log(softmax(score[y]))
    """
    score = run_IRNN(x)
    bsize, seq_len = x.shape
    return dy.sum_batches(dy.pickneglogsoftmax_batch(score, y)) / bsize


def get_probabilities(x, stats=False, noise=0.0):
    """
    Get probabilities for each class
    """
    score = run_IRNN(x)
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
        idx = range(i, min(i+opt.batch_size,N_dev))        # Get minibatch
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
        model.save(opt.model_out, [Wx_p, Wh_p, bh_p, A_p, b_p])
    trainer.update_epoch()                      # Update learning rate


def test():
    """
    Evaluate the model on the test data
    """
    start = time.time()                         # Time the iteration
    accuracy = 0
    for i in range(0, N_test, opt.batch_size):
        idx = range(i,  min(i+opt.batch_size,N_test))      # Get minibatch
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
