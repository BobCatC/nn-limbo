import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    assert predictions.ndim in [1, 2]

    if predictions.ndim == 1:
        exps = np.exp(predictions - np.max(predictions))
        return exps / np.sum(exps)
    else:
        exps = np.exp(predictions - np.max(predictions, axis=1).reshape(-1, 1))
        return exps / np.sum(exps, axis=1).reshape(-1, 1)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)

    if probs.ndim == 1:
        return -1 * np.log(probs[target_index])
    else:
        return -1 * np.sum(np.log(probs[np.arange(probs.shape[0]), target_index.flatten()])) / probs.shape[0]


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = probs.copy()

    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(predictions.shape[0]), target_index.flatten()] -= 1
        dprediction /= predictions.shape[0]

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return X * (X > 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return np.multiply(self.X >= 0, d_out)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

    def name(self):
        return "ReLU"


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.n_output = n_output
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Param(X)
        dot = np.dot(self.X.value, self.W.value)
        return dot + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        dx = np.dot(d_out, self.W.value.T)

        dw = self.X.value.T.dot(d_out)
        db = np.sum(d_out, axis=0)
        db = np.reshape(db, self.B.value.shape)

        self.X.grad = dx
        self.W.grad = dw
        self.B.grad = db

        d_input = dx

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    def name(self):
        return "FC-{}".format(self.n_output)

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - (self.filter_size - 1) + 2 * self.padding
        out_width = width - (self.filter_size - 1) + 2 * self.padding

        x_transformed = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, channels))
        x_transformed[:, self.padding: height + self.padding, self.padding: width + self.padding, :] = X

        self.X = X, x_transformed

        output = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_window = x_transformed[
                           :,
                           y: y + self.filter_size,
                           x: x + self.filter_size,
                           :,
                           np.newaxis
                        ]

                output[:, y, x, :] = np.sum(x_window * self.W.value, axis=(1, 2, 3)) + self.B.value

        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        X, X_with_padding = self.X

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        X_grad = np.zeros(X_with_padding.shape)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                window = X_with_padding[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                # print((grad*window).shape, self.W.grad.shape)
                # print(np.sum(grad * window, axis=0))
                self.W.grad += np.sum(grad * window, axis=0)
                # pass
                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=4)
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:height + self.padding, self.padding:width + self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def name(self):
        return "Conv{}x{}-{}".format(self.filter_size, self.filter_size, self.out_channels)


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, channels))

        self.X = X

        for y in range(out_height):
            for x in range(out_width):
                x_window = X[
                           :,
                           y: y + self.pool_size,
                           x: x + self.pool_size,
                           :
                        ]

                output[:, y, x, :] = np.max(x_window, axis=(1, 2))

        return output


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                x_window = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                dx = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]

                max_el = (x_window == np.max(x_window, axis=(1, 2))[:, np.newaxis, np.newaxis, :])

                output[:, y: y + self.pool_size, x: x + self.pool_size, :] += dx * max_el

        return output

    def params(self):
        return {}

    def name(self):
        return "MaxPooling{}x{}".format(self.pool_size, self.stride)


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

    def name(self):
        return "Flatten"
