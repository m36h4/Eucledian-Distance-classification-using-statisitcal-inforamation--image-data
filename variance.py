import csv
import random
import math
import operator
from pandas import *
import numpy as np
import time
start_time = time.time()
def main():
	var1 = np.zeros((24,24), dtype=float)
	var2 = np.zeros((24,24), dtype=float)
	var3 = np.zeros((24,24), dtype=float)
	var_atr = np.zeros((24,24), dtype=float)
	var_gen = np.zeros((24,24), dtype=float)
	var_obs = np.zeros((24,24), dtype=float)
	data=[]
	mean=[]
	#newList=[][] 
	with open('/home/megha/Desktop/Image data/out.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			mean.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	with open('/home/megha/Desktop/Image data/train.data', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = dataset[x][y]
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
			data.append(dataset[x])
	for col in range(0,24):
		for row in range(0,150):
			if row < 50:
				var1[col][col] += pow((float(data[row][col]) - float(mean[0][col])), 2)
				#print(var1[col][col])
			if row >= 50 and row < 100:
				var2[col][col] += pow((float(data[row][col]) - float(mean[1][col])), 2)
			if row > 100:
				var3[col][col] += pow((float(data[row][col]) - float(mean[2][col])), 2)
	#print('var1:')
	#print(var1)
	#print('var2:')
	#print(var2)
	#print('var3:')
	#print(var3)
	myInt =1/ 49
	var_atr = np.multiply(var1, myInt)
	var_gen = np.multiply(var2, myInt)
	var_obs = np.multiply(var3, myInt)
	print("var_atr")
	print(var_atr)
	print('var_gen')
	print(var_gen)
	print('var_obs')	
	print(var_obs)
main()
print("Time Taken :: --- %s seconds ---" % (time.time() - start_time))               
