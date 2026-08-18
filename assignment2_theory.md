# Assignment 2, Part 3: Arithmetic of CNNs

The output size of a convolution or pooling layer is

    output = floor((W - K + 2P) / S) + 1

where W is the input size, K the kernel size, P the padding, and S the stride.

## Question 1

Input 32 x 32 x 3, 8 filters of size 5 x 5, stride 1, no padding.

    output = floor((32 - 5 + 0) / 1) + 1 = 28

The number of filters sets the depth, so the output is

    28 x 28 x 8

## Question 2

With "same" padding the spatial size is kept equal to the input size. For a 5 x 5
kernel this needs P = 2.

    output = floor((32 - 5 + 4) / 1) + 1 = 32

So the output becomes

    32 x 32 x 8

## Question 3

Input 64 x 64, kernel 3 x 3, stride 2, no padding.

    output = floor((64 - 3 + 0) / 2) + 1 = floor(61 / 2) + 1 = 30 + 1 = 31

The output spatial size is

    31 x 31

## Question 4

Max pooling with kernel 2 x 2 and stride 2 on a 16 x 16 feature map.

    output = floor((16 - 2) / 2) + 1 = 8

The output is

    8 x 8

Pooling does not change the number of channels.

## Question 5

Input 128 x 128, two conv layers, each 3 x 3, stride 1, "same" padding.

Same padding keeps the spatial size, so each layer leaves it at 128 x 128.

    layer 1: 128 x 128
    layer 2: 128 x 128

The output shape is 128 x 128, with depth equal to the number of filters in the
second layer.

## Question 6

model.train() puts the model in training mode. A model is already in training mode
right after it is built, so on a first run nothing changes.

The problem appears once model.eval() has been called, for example to check accuracy
after an epoch. Without model.train() the model stays in evaluation mode, and two
layer types then behave differently:

    Dropout    turned off, so no regularisation happens
    BatchNorm  uses stored running statistics instead of batch statistics,
               and stops updating them

Training still runs, but the result is wrong or worse than it should be.

The CNN in this assignment has no dropout and no batch normalisation, so removing
the line makes no difference here. It is still good practice to keep it, since the
bug is silent and only appears after a layer of that kind is added.
