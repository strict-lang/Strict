# Strict.NeuralNetworks
A multilayer perceptron written in Strict, the use case for measuring parallel loops on all CPU cores, SIMD and the GPU. `Layer` is a fully connected layer (sigmoid, one weight row per neuron), `Network` trains any number of layers with backpropagation (softmax and cross entropy for several outputs, momentum 0.5), `Csv` and `Dataset` read samples, scale them to 0..1, one-hot encode the classes and hold every fifth sample back for testing.

Start with `Xor.strict` (2-3-1 neurons learn XOR in 11 ms natively), then `Iris.strict`: `strict build NeuralNetworks/Iris.strict` (from the repository root) trains 4-8-3 neurons for 200 epochs and reaches 97% test accuracy in 25 ms natively, 2.4 s on the Strict VM. `Diabetes.strict` learns a quarter of the UCI Diabetic Retinopathy Debrecen data (288 rows, 19 features), 60-64% in 80 ms.

Bigger data does not belong into this repository: the MNIST digits are in [Mnist](https://github.com/strict-lang/Mnist), larger datasets should be downloaded when they are needed.
