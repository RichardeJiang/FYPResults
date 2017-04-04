import numpy as np
import os
import math

def readFile(filePath, numOfSeq):
	fileContent = np.loadtxt(filePath)
	lengthOfFile = fileContent.size
	fileContent = fileContent.reshape(lengthOfFile/numOfSeq, numOfSeq)
	print(fileContent)
	return fileContent

def findNumOfSeq(filePath):
	fp = open(filePath, "r")
	lines = fp.readlines()
	words = lines[2].split()
	return int(words[1])

def writeToFile(filePath, matrix):
	fp = open(filePath, 'w')
	for x in np.nditer(matrix, op_flags=['readwrite']):
		x[...] = math.exp(x)
	(x, y) = matrix.shape
	print matrix
	np.savetxt(filePath, matrix)
	return

if (__name__=="__main__"):

	info_path = "lda-seq/info.dat"
	numOfSeq = findNumOfSeq(info_path)
	count = 0
	for file in os.listdir("lda-seq/"):
		if file.endswith("prob.dat"):
			dir_path = "lda-seq/" + file
			matrix = readFile(dir_path, numOfSeq)
			writeToFile("test" + str(count) + ".dat", matrix)
			count += 1
	pass