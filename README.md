# Neural Network

This library offer a framework to build neural network in plain numpy.

----
## USAGE

### NeuralNetwork

this class allow to define a neural network architecture by defining all the layers structure.
the parameters for the constructor are:
 * *eta* the learning rate,
 * *target_loss*, *target_epochs*, and *target_acc*, parameters to set the criteria to stop the trainig.
 * *mode*: it should be 'batch' if the weights are updated at the end of each epochs, 'on-line' if the weights must
    be update once for sample or 'mini-batch' if the weitghs are updated at fixed number of sample.    
 * *batch_size*: the size of the batch if mode = 'mini-batch'
 * *loss_fun*: loss function, should be 'sum-of-square' or 'cross-entropy' 

NeuralNetwork class is shipped with 3 main public methods:

* *addAll( layers )* : this method takes as input a list of layers object and add them to the net
* *predict( x )* : this method propagate the x vector in input and return network output
* *fit( X, y, valid_X, valid_y)* : this method allow the training of the network. returns a list of dict with info
  about each epoch. pressing crtl+c stop the training preserving the internal state of the network.
  if valid_X and valid_y are passed, they will be used to validate each epoch.
* *print()* : print the structure of the network
* *save( filepath )* save the net to filepath in pikle format
* *from_file(filepath)* static method that return a NeuralNetwork object loaded from a filepath
* *reset()*: reinitialize network

### LAYERS

Layers are the core of the neural network. a layer is a class that define
the following attributes:

 * tag: string name of the layer
 * shape: ouptut shape of the layer

it also have to define the following methods:

  * forw_prop( input ): takes as input an ndarray and return the output of the layer
  * back_prop( delta ): takes as input the gradient of error wrt the output of the layer,
    and returns the gradient of error wrt, the input of the layer
  * update( eta, mustupdate, batch_size): takes as input the eta (the value defined in the constructor of the net ),
    mustupdate, a boolean value that tell if the end of the batch is arrived ( always true in on-line) and
    the batch size. 
  * reset(): reset the weights of the net


NOTE ON ETA: the parameter eta in reality is very flexible and should be intended as the way
to pass hyperparameters to layers. common optimizer like sgd or gradient-descent have to know
only eta to work properly. other algoritm like rprop instead doesen't have something like learn rate but
might need other parameters. in resilient layers eta parameter is considered as the delta_zero of the network.
when we want to design a new network with new kind of layers we must think wich hyperparameters are layer-indipendent
and wich are constrained to be the same for all the layer in the network, those can be passed as eta .
if more then 1 parameter is required, we can use a dict/array/list as eta


## IMPLEMENTED LAYERS AND FUNCTIONS

this library provide a small set of ready to use activation function that can be used for net construction:

* tanh
* leaky_relu
* relu
* sigmoid
* identity
* softmax


there are 3 modules with already implemented layers

### neural_network.layers.dense_layers

* InputLayer(input_size): placeholder layer to input value in the net
* FullyConnectedLayer(input_size, output_size, actifun): hidden layer, input size must 
    fit the shape of the previous layer. actifun is the activation function

### neural_network.layers.convolution_layers

* PaddingLayer( pad ): add a padding to the image. 
  out_size = input_shape + 2 * padding
* PoolingLayer(out_size, mode): pooling layer, out_size is the shape of the output of previous
  layer. mode should be one "max" or  "avg". stride and window size are calculated using
  the indicated out_size
* ConvolutionalLayer(in_channels, n_filters, kernel_size, activation): convolution layer. the parameters are quite
  self-explainatory
  out_size= ( n_filters, img_size - kernel_size + 1)
* FlattenerLayer(in_channels, in_size): this layer should be beetween convolutional part of the net and dense
  part of the net, it will replace the inputlayer of the dense net. shape is in the format. in_channels is the
  number of channel in the output and in_size is the size of the image.
  output_size: in_channel * in_size * in_size


### neural_network.layers.resilient_layers

* ResilientFullyConnectedLayer(input_neuroids, n_neuroids, activation, dmax=50), dense layer with rprop optimizer
* ResilientConvolutionalLayer(in_channels, n_filters, kernel_size, activation, dmax=50), convolutional layer with
 rprop optimizer


---

### TEST ESAMPLES

the test folder contain 4 example experiments:


* test_conv.py: test a convolution network on sklearn digit dataset
* test_conv_resilient.py: same but with r-prop
* test_wine.py: test a mlp with the sklearn wine dataset
* test_wine_resilient.py: same but with r-prop
* test_mnist_2_3_7.py: test a cnn on mnist dataset
* test_mnist_res_2_3_7.py: same but with r-prop
* gui.py: a tkinter gui app that allow to paint digit and use the net trained on MNIST data to 
  classify them
