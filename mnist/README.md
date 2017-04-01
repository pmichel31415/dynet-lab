# MNIST Experiments

## Download the data


## Multilayer Perceptron

Results : 

python mnist/mlp.py -en test -ve 600 -te 600

98.31% accuracy

## IRNN

python mnist/irnn.py -en test -ve 100 -te 600 -dh 100 -lr 1e-7 -bs 16 -lrd 0.0 -ni 100000
71.18