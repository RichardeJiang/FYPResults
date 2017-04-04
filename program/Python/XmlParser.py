import os
from xml.dom.minidom import parse
import xml.dom.minidom
from Parser import XmlParser

def writeToFile(filePath, content):
	fp = open(filePath, 'w')
	fp.write(content.encode('ascii', 'ignore'))  #otherwise there will be unicode encoding problems
	fp.close()
	return

def parseXml(fileList):
	rootTag = "article_rec"
	targetTags = ["title", "par", "article_publication_date"]
	parser = XmlParser(fileList, rootTag, targetTags)
	return parser.parse()

if (__name__=='__main__'):
	dirList = os.listdir('.')
	fileList = []
	for fileName in dirList:
		if fileName.endswith('.xml'):
			fileList.append(fileName)

	procTitle = fileList[0].split('-')[0]
	areaTitle = fileList[0].split('-')[1]
	wholeList = []
	procList = []
	areaList = []
	for name in fileList:
		contents = name.split('-')
		proc = contents[0]
		area = contents[1]
		if (procTitle == proc):
			if (areaTitle == area):
				pass
			else:
				procList.append(areaList)
				areaList = []
				areaTitle = area
			areaList.append(name)
		else:
			procList.append(areaList)
			wholeList.append(procList)
			procList = []
			areaList = []
			areaTitle = area
			procTitle = proc

	count = 1

	# this is the part for consolidating all corpus into one single doc
	parsedValues = ""
	doc = "a/data_input.txt"
	for procs in wholeList:
		for areas in procs:
			if parseXml(areas).strip():
				parsedValues += parseXml(areas).strip()
	writeToFile(doc, parsedValues)

	# #this part is specifically for dividing the corpus according to conference names
	# #commented out temporarily 
	# for procs in wholeList:
	# 	#print item
	# 	for areas in procs:
	# 		parsedValues = ""
	# 		#doc = "data-" + areas[0].split('-')[0] + "-" + areas[0].split('-')[1] + ".txt"
	# 		doc = "testPy/data-" + str(count) + ".txt"
	# 		parsedValues = parseXml(areas)
	# 		if parsedValues.strip():
	# 			writeToFile(doc, parsedValues.strip())
	# 			count += 1
	pass