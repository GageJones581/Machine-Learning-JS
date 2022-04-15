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

    this.mw = math.matrix(math.zeros(outputs, inputs))
    this.vw = math.matrix(math.zeros(outputs, inputs))

    this.mb = math.matrix(math.zeros(outputs, 1))
    this.vb = math.matrix(math.zeros(outputs, 1))
  }

  forward (input) {
    if (input.constructor.name === "Matrix") this.input = input
    else this.input = math.matrix(input)
    this.output = math.add(math.multiply(this.weights, this.input), this.bias)
    return this.output
  }

  backward (outputGradient, learningRate, iter) {
    const weightsGradient = math.multiply(outputGradient, math.transpose(this.input))


    this.bias = math.subtract(this.bias, math.multiply(outputGradient, learningRate))
    this.weights = math.subtract(this.weights, math.multiply(weightsGradient, learningRate))

    const beta1 = 0.94
    const beta2 = 0.9878
    const eps = 10**-8

    for (let i = 0; i < this.weights._size[0]; i++) {
      for (let j = 0; j < this.weights._size[1]; j++) {
        this.mw._data[i][j] = beta1 * this.mw._data[i][j] + (1 - beta1) * weightsGradient._data[i][j]
        this.vw._data[i][j] = beta2 * this.vw._data[i][j] + (1 - beta2) * weightsGradient._data[i][j] ** 2
        const mhat = this.mw._data[i][j] / (1 - beta1 ** (iter + 1))
        const vhat = this.vw._data[i][j] / (1 - beta2 ** (iter + 1))
        this.weights._data[i][j] -= learningRate * mhat / (math.sqrt(vhat) + eps)
      }
    }

    for (let i = 0; i < this.bias._size[0]; i++) {
      for (let j = 0; j < this.bias._size[1]; j++) {
        this.mb._data[i][j] = beta1 * this.mb._data[i][j] + (1 - beta1) * outputGradient._data[i][j]
        this.vb._data[i][j] = beta2 * this.vb._data[i][j] + (1 - beta2) * outputGradient._data[i][j] ** 2
        const mhat = this.mb._data[i][j] / (1 - beta1 ** (iter + 1))
        const vhat = this.vb._data[i][j] / (1 - beta2 ** (iter + 1))
        this.bias._data[i][j] -= learningRate * mhat / (math.sqrt(vhat) + eps)
      }
    }

    const inputGradient = math.multiply(math.transpose(this.weights), outputGradient)
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

  backward (outputGradient, learningRate, iter) {
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
  let tempTrue = math.matrix(math.clone(yTrue))
  let tempPred = math.matrix(math.clone(yPred))

  if (tempTrue._size.length === 1) tempTrue = math.resize(tempTrue, [tempTrue._size[0], 1])
  if (tempPred._size.length === 1) tempPred = math.resize(tempPred, [tempPred._size[0], 1])

  return math.mean(math.square(math.subtract(tempTrue, tempPred)))
}

exports.msePrime = function (yTrue, yPred) {
  let tempTrue = math.matrix(math.clone(yTrue))
  let tempPred = math.matrix(math.clone(yPred))

  if (tempTrue._size.length === 1) tempTrue = math.resize(tempTrue, [tempTrue._size[0], 1])
  if (tempPred._size.length === 1) tempPred = math.resize(tempPred, [tempPred._size[0], 1])

  return math.divide(math.multiply(math.subtract(tempPred, tempTrue), 2), tempTrue._data.length)
}

const zip = (a, b) => a.map((k, i) => [k, b[i]])

exports.Network = class {
  constructor (inputs, outputs, numHiddenLayers = 2) {
    this.layers = []
    this.layers.push(new exports.Dense(inputs, outputs * (numHiddenLayers + 2)))
    this.layers.push(new exports.Tanh())

    let prevLayerOutputs = outputs * (numHiddenLayers + 2)
    for (let i = 0; i < numHiddenLayers; i++) {
      this.layers.push(new exports.Dense(prevLayerOutputs, outputs * (numHiddenLayers - i + 2)))
      prevLayerOutputs = outputs * (numHiddenLayers - i + 2)
      this.layers.push(new exports.Tanh())
    }

    this.layers.push(new exports.Dense(prevLayerOutputs, outputs))
    this.layers.push(new exports.Tanh())
  }

  setLayers (layers) {
    this.layers = layers
  }

  predict (input) {
    let temp = input

    let output

    if (temp.constructor.name !== 'Matrix') temp = math.matrix(temp)

    if (temp._size.length === 1) output = math.resize(temp, [temp._size[0], 1])
    else output = temp

    for (const layer of this.layers) {
      output = layer.forward(output)
    }
    return output
  }

  train (xTrain, yTrain, loss, lossPrime, learningRate, epochs = 1000, verbose = false) {
    const errors = []
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
          grad = layer.backward(grad, learningRate, e)
        }
        this.layers.reverse()
      }
      error /= zippedXY.length
      errors.push(error)

      if (verbose) console.log(`${e + 1}/${epochs}, error=${error}`)
    }
    return errors
  }
}
