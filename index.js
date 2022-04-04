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

  return math.divide(math.multiply(math.subtract(tempPred, tempTrue), 2), tempTrue.length)
}

const zip = (a, b) => a.map((k, i) => [k, b[i]])

exports.Network = class {
  constructor (inputs, outputs, numHiddenLayers = 2) {
    this.layers = []
    this.layers.push(new exports.Dense(inputs, outputs * (numHiddenLayers + 2)))
    this.layers.push(new exports.Tanh())

    //console.log('input : ', this.layers[0].weights._size)


    let prevLayerOutputs = outputs * (numHiddenLayers + 2)
    for (let i = 0; i < numHiddenLayers; i++) {
      this.layers.push(new exports.Dense(prevLayerOutputs, outputs * (numHiddenLayers - i + 2)))
      prevLayerOutputs = outputs * (numHiddenLayers - i + 2)
      this.layers.push(new exports.Tanh())

      //console.log(i, ':', this.layers[2 * i + 2].weights._size)
    }

    this.layers.push(new exports.Dense(prevLayerOutputs, outputs))
    this.layers.push(new exports.Tanh())


    //console.log('output : ', this.layers[this.layers.length - 2].weights._size)
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

  train (xTrain, yTrain, loss, lossPrime, learningRate, epochs = 1000, verbose=false) {
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
          grad = layer.backward(grad, learningRate)
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
