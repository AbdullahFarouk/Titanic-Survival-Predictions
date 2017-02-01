import csv
import time
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from math import sqrt
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from numpy.linalg import inv, lstsq


TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

#Define function to convert data into a format to allow operations to be performed on it

def getDataFromFile(fileName):
	data = []
	mode = 'rb'
	with open(fileName, mode) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data.append(row)

	# the first element is the header
	return data[1:]

#Define funcrion to create a matrix of features and a vector of labels
#Offset helps correct for column mismatch when reading in test data
#Concluded that (through Excel analysis) that the most important features are sex, Age, Number of sbilings on board, Fare paid and port embarked from

def makeMatrixAndLabels(data, forTraining = True):
	tempMatrix = []
	tempLabels = []
	offset = 0 if forTraining else 1

	for line in data:
		tempRow = []
		tempRow.append(1 if line[4 - offset] == 'male' else 0)
		if not line[5 - offset]:
			continue
		tempRow.append(float(line[5 - offset]))
		tempRow.append(float(line[6 - offset]))
		tempRow.append(1 if (line[9 - offset]) >= '20' else 0) 
		tempRow.append(1 if (line[11 - offset]) == 'S' else 0)
		tempMatrix.append(tempRow)

		print tempRow

		if not forTraining:
			continue

		tempLabels.append(int(line[1]))


	matrix = np.array(tempMatrix)

	if not forTraining:
		return matrix

	labels = np.array(tempLabels)
	return matrix, labels

data = getDataFromFile(TRAIN_FILE_NAME)
testData = getDataFromFile(TEST_FILE_NAME)
matrix, labels = makeMatrixAndLabels(data)

#Sampled with different split tests_size values (like 0.2,0.4, o.5, 0.6 and 0.8) and chose 0.4 to minimize overfitting

xTrain, xTest, yTrain, yTest = train_test_split(matrix, labels, test_size = 0.4, random_state = 100)

#Use logistic regression to predict passenger survival on board the titanic

regressor = LogisticRegression()
regressor.fit(xTrain, yTrain)
print regressor.coef_
print regressor.intercept_
acc = regressor.score(xTest, yTest)
print acc


