# 1301164545 - Azhari Rizkita P
import numpy
import math
import csv
import operator

# --------------- Driver --------------- #
def load(data):
    return numpy.genfromtxt(data, delimiter=',', skip_header=1)

def export(data):
    outputFile = open('1301164545_output.csv', 'wt', newline ='')
    header = ['Index','X1','X2','X3','X4','X5','Y']
    with outputFile:
        writer = csv.writer(outputFile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)
# --------------- Driver --------------- #

# --------------- Euclidean --------------- #
def euclidean(train, test):
    distance = 0
    for x in range(1,6):
        distance += pow((test[x] - train[x]), 2)
    return math.sqrt(distance)
# --------------- Euclidean --------------- #

# --------------- K-nearest-neighbor --------------- #
def knn(dataTrain, dataTest, k):
    neighbors = {}

    for x in range(len(dataTest)):
        distances = {}

        for y in range(len(dataTrain)):
            distances[y] = (euclidean(dataTrain[y], dataTest[x]))

        distances = sorted(distances.items(), key=operator.itemgetter(1))
        neighbors[x] = [distances[i] for i in range(k)]

    prediction = {}
    for x in neighbors:
        classVote = {}

        for data in neighbors[x]:
            value = dataTrain[data[0]][6]

            if value in classVote:
                classVote[value] += 1
            else :
                classVote[value] = 1
        
        prediction[x] = max(classVote.items(), key=operator.itemgetter(1))[0]
    
    return prediction
# --------------- K-nearest-neighbor --------------- #

# --------------- Main --------------- #
if __name__ == "__main__":
    data_file = load('train.csv')
    dataTrain = [data_file[x] for x in range(600)]
    dataTest = [data_file[x] for x in range(600, 800)]

    k = 5
    kFinal = k
    maximum = 0

    # Validation split
    while k < 50:
        prediction = knn(dataTrain, dataTest, k)
        result = 0
        for x in range(len(dataTest)):
            if (prediction[x] == dataTest[x][6]):
                result += 1
        if (result > maximum):
            maximum = result
            kFinal = k
        k += 2

    dataTest = load('test.csv')
    dataTrain = load('train.csv')

    # Mengisi nilai Y
    prediction = knn(dataTrain, dataTest, kFinal)
    for x in range(len(dataTest)):
        dataTest[x][6] = prediction[x]

    export(dataTest)
# --------------- Main --------------- #