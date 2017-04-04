import os
import numpy as np

def sortDict(inputList):
	inputList.sort(key=lambda line: int(line[0]))

def writeToNewDict(filePath, matrix):
	fp = open(filePath, 'w')
	for line in matrix:
		fp.write("%s" % ','.join(line))

def readDict(filePath):
	fp = open(filePath, 'r')
	lines = fp.readlines()
	contents = []
	for line in lines:
		content = line.split(',')
		contents.append(content)

	return contents

if (__name__=="__main__"):
	
	matrix = readDict("dict.txt")
	sortDict(matrix)
	filePath = 'dict.txt.new'
	writeToNewDict(filePath, matrix)

	pass