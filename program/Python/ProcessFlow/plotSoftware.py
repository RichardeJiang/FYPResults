import numpy as np
import matplotlib.pyplot as plt
import os

def readFile(filePath):
	fp = open(filePath, 'r')
	lines = fp.readlines()
	matrix = []
	for line in lines:
		lineContent = line.split(',')
		matrix.append(lineContent)
	return matrix

def plotMatrix(matrix, index, startingYear):
	size = len(matrix[0]) - 2
	# X = range(0, size)
	X = range(startingYear, startingYear + size)
	plt.xlabel('year')
	plt.ylabel('frequency')
	plt.xlim(startingYear, startingYear + size + 6)
	plt.title('Topic No. ' + str(index))
	for line in matrix:
		temp = line[2 : size+2]
		Y = [float(i) for i in temp]
		#plt.plot(X, Y, label=line[0])
		plt.plot(X, Y, label=str(line[0]))
		plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize='small')
	plt.savefig(str(index) + "-top.png")
	#plt.show()
	plt.close()
	
	return

if (__name__=="__main__"):
	startingYear = 1958

	for index in range(0,20):
		path = 'test' + str(index) + 'top20words.dat'
		matrix = readFile(path)
		plotMatrix(matrix, index, startingYear)

	pass