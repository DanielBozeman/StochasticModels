import numpy as np
import matplotlib.pyplot as plt
import math
import brownianPathGenerator
import stochasticMethods
from stochasticMethods import SDEModel

def OUProcess():

    theta = 0.7
    mu = 1.5
    sigma = 0.06

    timeStep = pow(2, -10)

    timeInterval = [0,7]

    numSims = 10

    constants = [theta, mu, sigma]

    def alphaFunction(model, value, time):
        return (model.constantsList[0] * (model.constantsList[1] - value))
    
    def betaFunction(model, value, time):
        return model.constantsList[2]
    
    model = SDEModel(constants=constants)

    model.alphaFunction = alphaFunction
    model.betaFunction = betaFunction

    for i in range(numSims):
        times, path = brownianPathGenerator.makePath(timeInterval[0], timeInterval[1], timeStep)

        approximation = stochasticMethods.eulerMaruyama(model, 0, times, path)

        plt.plot(times, approximation)

    plt.show()

#Creates the appropriate SDEModel objects for the Heston and returns them
def hestonModel(interestRate : float, longVariance : float, reversionRate : float, volOfVol : float):
    #This section defines the vaiance model
    #################################################################################
    varianceConstants = [longVariance, reversionRate, volOfVol]

    varianceModel = SDEModel(constants=varianceConstants)

    def varianceAlpha(model : SDEModel, value : float, time : float):
        return (model.constantsList[1] * (model.constantsList[0] - value))

    def varianceBeta(model : SDEModel, value : float, time : float):
        return (model.constantsList[2] * math.sqrt(value))
    
    varianceModel.alphaFunction = varianceAlpha
    varianceModel.betaFunction = varianceBeta
    #################################################################################

    #This sections defines the stock model
    #################################################################################
    stockConstants = [interestRate]

    stockModel = SDEModel(constants=stockConstants)

    def stockAlpha(model : SDEModel, value : float, time : float):
        return(model.constantsList[0] * value)
    
    def stockBeta(model : SDEModel, value : float, time : float, volatility : float):
        return(math.sqrt(volatility) * value)
    
    stockModel.alphaFunction = stockAlpha
    stockModel.betaFunction = stockBeta
    #################################################################################

    return stockModel, varianceModel

def runHeston(initialValue : float, initialVariance : float, stockModel : SDEModel, varianceModel : SDEModel):

    interval = [0,1]
    
    timeStep = pow(2, -10)

    correlation = 0

    for i in range(10):
        times, path1, path2 = brownianPathGenerator.makeCorrelatedPaths(interval[0], interval[1], timeStep, correlation)

        varianceApproximation = stochasticMethods.eulerMaruyama(varianceModel, initialVariance, times, path2)

        stockApproximation = stochasticMethods.eulerMaruyamaStochasticVol(stockModel, initialValue, times, path1, varianceApproximation)

        plt.plot(times, stockApproximation)

    plt.show()
    

if __name__ == "__main__":

    interestRate = 0.1

    initialValue = 100
    
    reversionRate = 3.0
    longVariance = 0.2
    volOfVol = 0.1
    initialVariance = 0.1

    stockModel, varianceModel = hestonModel(interestRate, longVariance, reversionRate, volOfVol)

    runHeston(initialValue, initialVariance, stockModel, varianceModel)