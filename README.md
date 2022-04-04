# Machine-Learning-JS
Machine Learning JS is a node module designed to be a simple and easy to use library so that you don't have to know all of how training and optimization works to create a simple AI.
## Installation
First, you need to have node and npm installed. Once you have that installed, go to the project directory and run `npm install machinelearningjs`.
## Syntax
I'll move this to a jsdoc eventually.

`const ml = require('machinelearningjs')`

To create a neural network, use `const NETWORKNAME = ml.network(NUMINPUTS(int), NUMOUTPUTS(int), NUMHIDDENLAYERs(int))`. 

To train the neural network, use `const errors = NETWORKNAME.train(XTRAIN(arr), YTRAIN(arr), LOSS(func), LOSSPRIME(func), LEARNINGRATE(float), EPOCHS(int), VERBOSE(bool))`. This returns a list of all the errors from each epoch.

To receive a prediction, use `NETWORKNAME.predict(INPUT(arr))`.
