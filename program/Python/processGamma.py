import numpy as np
import os
import math

def readFile(filePath, numOfSeq):
	fileContent = np.loadtxt(filePath)
	lengthOfFile = fileContent.size
	#fileContent = fileContent.reshape(lengthOfFile/numOfSeq, numOfSeq)
	fileContent = fileContent.reshape(lengthOfFile/8, 8)
	print(fileContent)
	return fileContent

def findNumOfSeq(filePath):
	fp = open(filePath, "r")
	lines = fp.readlines()
	words = lines[2].split()
	return int(words[1])

def writeToFile(filePath, matrix):
	# np.savetxt(filePath, matrix);
	fp = open(filePath, 'w')
	#rowSums = np.sum(matrix, axis=1)
	matrix = matrix/matrix.sum(axis=1)[:,None]
	print matrix
	np.savetxt(filePath, matrix)
	return

if (__name__=="__main__"):
	#dir_path = os.path.dirname(os.path.realpath(__file__));
	dirList = []
	for dirName in os.listdir('.'):
		if dirName.endswith('0'):
			dirList.append(dirName)
			print(dirName)

	for dirName in dirList:
		info_path = dirName + "/lda-seq/info.dat"
		gamma_path = dirName + "/lda-seq/gam.dat"
		numOfSeq = findNumOfSeq(info_path)
		matrix = readFile(gamma_path, numOfSeq)
		writeToFile(dirName + "/gammaResult.dat", matrix)
	pass