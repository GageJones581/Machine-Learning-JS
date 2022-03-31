const math = require('mathjs')

exports.Layer = class {
  constructor () {
    this.input = null
    this.output = null
  }

  forward (input) { }

  backward (outputGradient, learningRate) { }
}

exports.Dense = class extends exports.Layer {
  constructor (inputs, outputs) {
    super()
    this.weights = math.matrix(math.ones(outputs, inputs))
    this.weights = this.weights.map(function (value, index, matrix) {
      return value * (math.random() * 2 - 1)
    })
    this.bias = math.matrix(math.ones(outputs, 1))
    this.bias = this.bias.map(function (value, index, matrix) {
      return value * (math.random() * 2 - 1)
    })
  }

  forward (input) {
    if (input.constructor.name === "Matrix") this.input = input
    else this.input = math.matrix(input)
    this.output = math.add(math.multiply(this.weights, this.input), this.bias)
    return this.output
  }

  backward (outputGradient, learningRate) {

    const weightsGradient = math.multiply(outputGradient, math.transpose(this.input))
    this.bias = math.subtract(this.bias, math.multiply(outputGradient, learningRate))
    const inputGradient = math.multiply(math.transpose(this.weights), outputGradient)
    this.weights = math.subtract(this.weights, math.multiply(weightsGradient, learningRate))
    return inputGradient
  }
}

exports.Activation = class extends exports.Layer {
  constructor (activation, activationPrime) {
    super()
    this.activation = activation
    this.activationPrime = activationPrime
  }

  forward (input) {
    this.input = input
    this.output = this.activation(input)
    return this.output
  }

  backward (outputGradient, learningRate) {
    const inputGradient = math.dotMultiply(outputGradient, this.activationPrime(this.input))
    return inputGradient
  }
}

exports.Tanh = class extends exports.Activation {
  constructor () {
    const activation = x => math.tanh(x)
    const activationPrime = x => math.subtract(1, math.square(math.tanh(x)))
    super(activation, activationPrime)
  }
}

exports.mse = function (yTrue, yPred) {
  return math.mean(math.square(math.subtract(yTrue, yPred)))
}

exports.msePrime = function (yTrue, yPred) {
  return math.divide(math.multiply(math.subtract(yPred, yTrue), 2), yTrue.length)
}

const zip = (a, b) => a.map((k, i) => [k, b[i]])

exports.Network = class {
  constructor (...layers) {
    this.layers = layers
  }

  predict (input) {
    let output = input
    for (const layer of this.layers) {
      output = layer.forward(output)
    }
    return output
  }

  train (xTrain, yTrain, loss, lossPrime, learningRate, epochs = 1000) {
    const zippedXY = zip(xTrain, yTrain)
    for (let e = 0; e < epochs; e++) {
      let error = 0
      for (let i = 0; i < zippedXY.length; i++) {
        const x = math.resize(zippedXY[i][0], [zippedXY[i][0].length, 1])
        const y = math.resize(zippedXY[i][1], [zippedXY[i][1].length, 1])

        const output = this.predict(x)
        let grad = lossPrime(y, output)

        error += loss(y, output)

        for (const layer of this.layers.reverse()) {
          grad = layer.backward(grad, learningRate)
        }
        this.layers.reverse()

      }
      error /= zippedXY.length
      console.log(`${e+1}/${epochs}, error=${error}`)
    }
  }
}

const X = [[0, 0], [0, 1], [1, 0], [1, 1]]
const Y = [[0], [1], [1], [0]]

const network = new exports.Network(
  new exports.Dense(2, 2),
  new exports.Tanh(),
  new exports.Dense(2, 1),
  new exports.Tanh()
)

network.train(X, Y, exports.mse, exports.msePrime, 0.1, 1000)

for (let i = 0; i < X.length; i++) {
  console.log(`Y: ${Y[i][0]}, Y*: ${network.predict(math.resize(X[i], [2, 1]))._data}`)
}
