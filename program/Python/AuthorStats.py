from xml.dom.minidom import parse
import xml.dom.minidom
import operator
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from itertools import takewhile, tee, izip, chain
import os
import math
import numpy as np

import networkx
import string
import sys

#added for solving the headache ascii encode/decode problem
reload(sys)  
sys.setdefaultencoding('utf8')

def writeListToFile(listFile, fileName):
	theFile = open(fileName, 'w')
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		for inner in item[1]:
			theFile.write("%s, " % str(inner))
		theFile.seek(-2, os.SEEK_CUR)
		theFile.write("\n")
	theFile.close()
	return

def writeFreqToFile(listFile, fileName):
	theFile = open(fileName, "w")
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		theFile.write("%s\n" % str(item[1]))
	return

def XmlParsing(targetFile, targetTag):
	try:
		DOMTree = xml.dom.minidom.parse(targetFile)
	except xml.parsers.expat.ExpatError, e:
		print "The file causing the error is: ", fileName
		print "The detailed error is: %s" %e
	else:
		collection = DOMTree.documentElement

		resultList = collection.getElementsByTagName(targetTag)
		return resultList

	return "ERROR"

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
	stop_words = set(nltk.corpus.stopwords.words('english'))
	tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
	candidates = [word.lower() for word, tag in tagged_words
		if tag in good_tags and word.lower() not in stop_words and len(word) > 1]
	return candidates

def getKeyphraseByTextRank(text, n_keywords=0.05, n_windowSize=2, n_cooccurSize=2):
	words = [word.lower()
		for word in nltk.word_tokenize(text)
		if len(word) > 1]
	
	candidates = extract_candidate_words(text)
	# print candidates
	graph = networkx.Graph()
	graph.add_nodes_from(set(candidates))
	
	for i in range(0, n_windowSize-1):
		def pairwise(iterable):
			a, b = tee(iterable)
			next(b, None)
			for j in range(0, i):
				next(b, None)
			return izip(a, b)
		for w1, w2 in pairwise(candidates):
			if w2:
				graph.add_edge(*sorted([w1, w2]))

	ranks = networkx.pagerank(graph)
	if 0 < n_keywords < 1:
		n_keywords = int(round(len(candidates) * n_keywords))
	word_ranks = {word_rank[0]: word_rank[1]
		for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	keywords = set(word_ranks.keys())

	keyphrases = {}
	j = 0
	for i, word in enumerate(words):
		if i<j:
			continue
		if word in keywords:
			kp_words = list(takewhile(lambda x: x in keywords, words[i:i+n_cooccurSize]))
			avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
			keyphrases[' '.join(kp_words)] = avg_pagerank

			j = i + len(kp_words)

	results = [(ele[0], ele[1]) for ele in sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)]
	# results = self.duplicateHigherRankingTerms(results)
	# targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
	# for result in results:
	# 	tempSet = self.removeDuplicates(result.split())
	# 	if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
	# 		results.remove(result)
	# 	else:
	# 		newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
	# 		results[results.index(result)] = newPhrase
	# results = self.removeDuplicates(results)
	# if (len(results) > 200):
	# 	results = results[:len(results) * 0.25]
	# return ' '.join(results)
	#results = {item[0]:item[1] for item in results}
	return duplicateHigherRankingTerms(results)

def duplicateHigherRankingTerms(rawList):
	rawList = removeDuplicates(rawList)
	if len(rawList) < 1:
		return ""
	baseFreq = float(rawList[-1][1])
	result = []

	phraseScoreMap = {}
	targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
	stemmer = SnowballStemmer("english")
	for ele in rawList:
		tempSet = removeDuplicates(ele[0].split())
		if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
			pass
		else:
			newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
			result.append((newPhrase, ele[1]))

	if (len(result) > 200):
		result = result[:len(result) / 4]

	phraseScoreMap = {item[0]:item[1] for item in result}

	return phraseScoreMap

def removeDuplicates(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def HITS(phraseScoreList, phraseAuthorMap, authorScoreList, authorPhraseMap):
	for count in range(0, 10000):
		norm = 0.0
		for author in authorPhraseMap:
			currPhraseList = authorPhraseMap[author]
			authorScore = 0
			for phrase in currPhraseList:
				authorScore += phraseScoreList[phrase]

			norm += (authorScore ** 2)
			authorScoreList[author] = authorScore
		norm = math.sqrt(norm)
		for author in authorScoreList:
			authorScoreList[author] = authorScoreList[author] / norm

		norm = 0.0
		for phrase in phraseAuthorMap:
			currAuthorList = phraseAuthorMap[phrase]
			phraseScore = 0
			for author in currAuthorList:
				phraseScore += authorScoreList[author]

			norm += (phraseScore ** 2)
			phraseScoreList[phrase] = phraseScore
		norm = math.sqrt(norm)
		for phrase in phraseScoreList:
			phraseScoreList[phrase] = phraseScoreList[phrase] / norm

	return phraseScoreList, phraseAuthorMap, authorScoreList, authorPhraseMap

if (__name__ == '__main__'):
	fileList = os.listdir('.')
	targetList = []
	for fileName in fileList:
		if fileName.endswith('.xml'):
			targetList.append(fileName)

	scoreList = {}
	authorIdMap = {}
	count = 0
	flag9000_1 = False
	flag9000_2 = False

	predefinedNumberOfVIPAuthors = 300
	predefinedNumberOfVIPPhrases = 300
	commonPhrasesRecognitionCriteria = int(0.25 * predefinedNumberOfVIPAuthors)

	for target in targetList:
		authorList = XmlParsing(target, "au")
		if authorList == "ERROR":
			continue

		# This part is to check whether all files have been iterated through
		count += 1
		if count > 9000:
			flag9000_1 = True

		for author in authorList:
			try:
				firstName = author.getElementsByTagName("first_name").item(0).childNodes[0].data
				lastName = author.getElementsByTagName("last_name").item(0).childNodes[0].data
				id = author.getElementsByTagName("author_profile_id").item(0).childNodes[0].data
			#print "first name is: " + str(firstName) + " last name is: " + str(lastName) + " " + id
			except IndexError, e:
				print "Xml file author tag parsing error: %s" %e
				#continue
			except AttributeError, e1:
				print "No selected attribute detected: %s" %e1
				#continue
			else:
				fullName = firstName.split(".")[0] + " " + lastName

				if not scoreList.has_key(id):
					scoreList[id] = 1
				else:
					scoreList[id] += 1

				if not authorIdMap.has_key(id):
					authorIdMap[id] = fullName
				else:
					if len(authorIdMap[id]) < len(fullName):
						authorIdMap[id] = fullName

	print 'Total number of authors is: ' + str(len(authorIdMap))
	sorted_scoreList = sorted(scoreList.items(), key = operator.itemgetter(1), reverse = True)[:predefinedNumberOfVIPAuthors]
	sorted_scoreDict = {}
	for item in sorted_scoreList:
		sorted_scoreDict[item[0]] = item[1]
	
	# Fundamental logic of HITS Algorithm
	# every doc, get keyphrases, with initial score of TextRankScore x sum(authorScore)
	# authorScore update according to normalized sum of keyphrases score ... 1
	# phraseScore update according to normalized sum of author score ... 2
	# iterate through 1 and 2

	# This part is to store the author score list computed above into an external file
	# np_sorted_author_scoreList = np.asarray(sorted_scoreList)
	# np_author_id_list = np.asarray([(k, v) for k, v in authorIdMap.items()])
	# np.savetxt('sorted_author_scoreList.txt', np_sorted_author_scoreList)
	# np.savetxt('author_id_map.txt', np_author_id_list)
	# print "Author score info saved."
	# End of the storing to file logic

	# This part is to read the stored file to restore the authorScoreList and AuthorIdMap
	# tempAuthorScoreList = np.loadtxt('sorted_author_scoreList.txt')
	# tempAuthorIdList = np.loadtxt('author_id_map.txt')
	# sorted_scoreDict = {k:float(v) for [k,v] in tempAuthorScoreList}
	# authorIdMap = {k:v for [k, v] in tempAuthorIdList}
	# End of txt file reading logic

	authorMap = [item for item in sorted_scoreDict]

	authorPhraseMap = {author:[] for author in sorted_scoreDict}
	phraseScoreList = {}
	phraseAuthorMap = {}

	count = 0
	for target in targetList:
		articleList = XmlParsing(target, "article_rec")
		if articleList == "ERROR":
			continue

		# This part is to check whether all files have been iterated through
		count += 1
		if count > 9000:
			flag9000_2 = True

		for article in articleList:
			authors = article.getElementsByTagName("author_profile_id")
			currAuthorMap = [item.childNodes[0].data for item in authors]
			if (len(set(currAuthorMap) & set(authorMap)) > 0):
				abstract = article.getElementsByTagName("ft_body")
				
				if len(abstract) > 0:
					abstract = abstract.item(0).childNodes[0].data
					abstract = re.sub(r'<.*?>', "", abstract)
					abstract = re.sub(r'\"', "", abstract)
					abstract = str(abstract.encode('utf-8')).translate(None, string.punctuation)
					abstract = ''.join([i for i in abstract if not i.isdigit()])

					currPhraseScoreMap = getKeyphraseByTextRank(abstract)
					currPhraseSet = set([currPhrase for currPhrase in currPhraseScoreMap])

					sumOfCurrAuthorScore = 0
					VIPAuthorSet = set(currAuthorMap) & set(authorMap)
					for currAuthor in list(VIPAuthorSet):
						sumOfCurrAuthorScore += sorted_scoreDict[currAuthor]
						authorPhraseMap[currAuthor] = list(set(authorPhraseMap[currAuthor]) | currPhraseSet)

					for currPhrase in currPhraseScoreMap:
						if not phraseScoreList.has_key(currPhrase):
							phraseScoreList[currPhrase] = currPhraseScoreMap[currPhrase] * sumOfCurrAuthorScore
						else:
							phraseScoreList[currPhrase] += currPhraseScoreMap[currPhrase] * sumOfCurrAuthorScore

						if not phraseAuthorMap.has_key(currPhrase):
							phraseAuthorMap[currPhrase] = list(VIPAuthorSet)
						else:
							temp = list(set(phraseAuthorMap[currPhrase]) | VIPAuthorSet)
							phraseAuthorMap[currPhrase] = temp

	phraseScoreList, phraseAuthorMap, sorted_scoreDict, authorPhraseMap = HITS(phraseScoreList, phraseAuthorMap, sorted_scoreDict, authorPhraseMap)

	newPhraseAuthorMap = {k:v for k, v in phraseAuthorMap.items() if len(v) < commonPhrasesRecognitionCriteria}
	phraseAuthorMap = newPhraseAuthorMap
	validPhraseCheckList = [item for item in phraseAuthorMap]
	
	sorted_phraseList = sorted(phraseScoreList.items(), key = operator.itemgetter(1), reverse = True)
	sorted_authorList = sorted(sorted_scoreDict.items(), key = operator.itemgetter(1), reverse = True)

	sorted_phraseListNoScore = [item[0] for item in sorted_phraseList]
	sorted_authorListNoScore = [item[0] for item in sorted_authorList]

	writeFreqToFile(sorted_phraseList[:predefinedNumberOfVIPPhrases], 'b/sorted_phraseList.txt')

	authorNamePhraseList = []
	for authorScore in sorted_authorList:
		author = authorScore[0]
		authorName = str(authorIdMap[author].encode('utf-8')).translate(None, string.punctuation)
		authorName = ''.join([i for i in authorName if not i.isdigit()])
		authorPhraseMap[author] = list(set(authorPhraseMap[author]) & set(validPhraseCheckList))
		if len(authorPhraseMap[author]) > 50:
			# Important note here: list a.sort(key = lambda xxxx): this statement returns no value!
			authorPhraseMap[author].sort(key=lambda x: sorted_phraseListNoScore.index(x))
			authorNamePhraseList.append((authorName, authorPhraseMap[author][:50]))
		else:
			authorNamePhraseList.append((authorName, authorPhraseMap[author]))

	phraseAuthorNameList = []
	for phraseScore in sorted_phraseList:
		phrase = phraseScore[0]
		if phrase in validPhraseCheckList:
			tempAuthorNameList = []
			phraseAuthorMap[phrase].sort(key=lambda x: sorted_authorListNoScore.index(x))
			if len(phraseAuthorMap[phrase]) > 30:
				phraseAuthorMap[phrase] = phraseAuthorMap[author][:30]
			for authorId in phraseAuthorMap[phrase]:
				authorName = str(authorIdMap[authorId].encode('utf-8')).translate(None, string.punctuation)
				authorName = ''.join([i for i in authorName if not i.isdigit()])
				tempAuthorNameList.append(authorName)
			phraseAuthorNameList.append((phrase, tempAuthorNameList))
	# authorNamePhraseMap = {}
	# for author in authorPhraseMap:
	# 	if len(authorPhraseMap[author]) > 20:
	# 		authorNamePhraseMap[authorIdMap[author]] = authorPhraseMap[author][:20]
	# 	else:
	# 		authorNamePhraseMap[authorIdMap[author]] = authorPhraseMap[author]
	writeListToFile(authorNamePhraseList, 'b/authorNamePhraseList.txt')
	writeListToFile(phraseAuthorNameList, 'b/phraseAuthorNameList.txt')

	if flag9000_1:
		print "1st 9000 has been reached"

	if flag9000_2:
		print "2nd 9000 has been reached"

	pass