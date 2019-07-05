import csv
import random
import math
import operator
import csv
#from sklearn.metrics import confusion_matrix, accuracy_score
#from sklearn.model_selection import cross_val_score
from pandas import *
#def loadDataset(filename, split, trainingSet=[] , testSet=[]):

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((int(instance1[x]) - int(instance2[x])), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	# prepare data
	trainingSet=[]
	testSet=[]
	#split = 0.67
	#loadDataset('/home/megha/Desktop/train.data', split, trainingSet, testSet)
	with open('/home/megha/Desktop/train.data', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			trainingSet.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	with open('/home/megha/Desktop/test.data', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
			testSet.append(dataset[x])
	#print 'Train set: ' + repr(len(trainingSet))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = input("Enter value of k: ") 
	print(k) 
	mean = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] 
	mean[0][24] = 'atrium_public'
	mean[1][24] = 'general_store_outdoor'
	mean[2][24] = 'observatory_outdoor'
	for col in range(0,24):
		a = 0
		b = 0
		c = 0
		for row in range(0,150):
			if y == 23:
				break
			#if trainingSet[row][-1] =='atrium_public':
			if row >= 0 and row < 50:
				a = a + int(trainingSet[row][col])
				#print(a)
			if row >= 50 and row < 100:
			#if trainingSet[row][-1] =='general_store_outdoor':
				b = b + int(trainingSet[row][col])
				#print(b)
			if row >= 100:
			#if trainingSet[row][-1] =='observatory_outdoor':
				c = c + int(trainingSet[row][col])
				#print(c)
		mean[0][col] = float(a)/50.0  
		mean[1][col] = float(b)/50.0 
		mean[2][col] = float(c)/50.0 
	#print(mean[0][5])
	#print(mean[1][5])
	#print(mean[2][5])
	for x in range(0,3):
		for y in range(0,25):
			print(mean[x][y])
	#for x in range(len(testSet)):
		#neighbors = getNeighbors(trainingSet, testSet[x], k)
		#result = getResponse(neighbors)
		#predictions.append(result)
		
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	#accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')	
	#print 'Confusion Matrix:'
	#print DataFrame(cm, columns=['bayou', 'music_store', 'desert_vegetation'], index=['bayou', 'music_store', 'desert_vegetation'])

	with open("out.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(mean)

main()
