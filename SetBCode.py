import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



### Task Set B ###


matrixSize = 3

matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
stateVectorOne = [1, 0, 0, 0]
stateVectorTwo = [0.15, 0.85, 0, 0]
setAProbMatrix = [[0.7, 0.4, 0, 0.2], [0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0.3, 1, 0.8]]


result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def createEmptySquareMatrix(matrixSize : int):
    matrix = [[0 for n in range(matrixSize)] for n in range(matrixSize)]
    return matrix



def matrixMultiplication(matrixSize : int, matrix1 : list, matrix2 : list):
    result = createEmptySquareMatrix(matrixSize)
    for i in range(matrixSize):
        for j in range(matrixSize):
            currentSpot = 0
            for k in range(matrixSize):
                currentSpot += matrix1[i][k] * matrix2[k][j]
            result[i][j] = currentSpot

    return result



def matrixToPower(matrixSize : int, matrix : list, power : int):
    result = matrix
    for n in range(power):
        result = matrixMultiplication(matrixSize, result, matrix)

    return result



def findProbability(matrixSize : int, transitionMatrix : list, stateVector : list):
    result = [0 for n in range(matrixSize)]
    for i in range(matrixSize):
        currentValue = 0
        for j in range(matrixSize):
            currentValue += transitionMatrix[i][j] * stateVector[j]
        result[i] = currentValue

    return result

def getNormalizedVector(vecLength : int, vector : list) -> list:
    
    norm = 0
    newVector = vector
    
    for x_i in newVector: 
        norm += x_i**2
    
    norm = math.sqrt(norm)
    
    for i in range(vecLength): 
        newVector[i] = newVector[i] / vecLength
    
    return newVector


def plotProbability(matrixSize : int, numSteps : int, transitionMatrix : list, stateVector : list, title):
    plt.figure()
    plotLegend = ["Susceptible", "Exposed", "Infected", "Recovered"]
    currentProbVector = [0 for n in range(matrixSize)]
    allProbVectors = []
    currentProbMatrix = transitionMatrix ### createEmptySquareMatrix(matrixSize)
    xVector = [(n + 1) for n in range(numSteps - 1)]
    
    for i in range(numSteps - 1):
        currentProbMatrix = matrixMultiplication(matrixSize, currentProbMatrix, transitionMatrix)
        currentProbVector = findProbability(matrixSize, currentProbMatrix, stateVector)
        
        
        ### Normalize Vectors -- Doesn't seem to be needed, or working ###
       
        ######################################################
        allProbVectors.append(getNormalizedVector(matrixSize, currentProbVector))  ### allProbVectors.append(iterVector)
    
    for v in range(matrixSize):
        yVector = []
        for w in range(numSteps - 1):
            yVector.append(allProbVectors[w][v])
            ### print(allProbVectors[w-1][v]) ###
        plt.plot(xVector, yVector, label = plotLegend[v])
        plt.title(title)
        plt.legend()

### Problem 1 ###
plotProbability(4, 31, setAProbMatrix, stateVectorOne, "Problem 1")



### Problem 2 ###
plotProbability(4, 31, setAProbMatrix, stateVectorTwo, "Problem 2")

plt.show()