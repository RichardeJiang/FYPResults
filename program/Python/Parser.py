from xml.dom.minidom import parse
import xml.dom.minidom
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from itertools import takewhile, tee, izip, chain

import networkx
import string
import sys

#added for solving the headache ascii encode/decode problem
reload(sys)  
sys.setdefaultencoding('utf8')

class XmlParser:
	''' The general-purpose Xml parser '''

	def __init__(self, fileList, rootTag, targetTags):
		self.fileList = fileList
		self.rootTag = rootTag
		self.targetTags = targetTags
		self.content = ""

	def extract_candidate_words(self, text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
		stop_words = set(nltk.corpus.stopwords.words('english'))
		tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
		candidates = [word.lower() for word, tag in tagged_words
			if tag in good_tags and word.lower() not in stop_words and len(word) > 1]
		return candidates

	def getKeyphraseByTextRank(self, text, n_keywords=0.2, n_windowSize=2, n_cooccurSize=2):
		words = [word.lower()
			for word in nltk.word_tokenize(text)
			if len(word) > 1]
		
		candidates = self.extract_candidate_words(text)
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
		return self.duplicateHigherRankingTerms(results)

	def duplicateHigherRankingTerms(self, rawList):
		rawList = self.removeDuplicates(rawList)
		if len(rawList) < 1:
			return ""
		baseFreq = float(rawList[-1][1])
		result = []
		targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
		stemmer = SnowballStemmer("english")
		for ele in rawList:
			tempSet = self.removeDuplicates(ele[0].split())
			if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
				pass
			else:
				newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
				mult = int(float(ele[1]) / baseFreq)
				for i in range(0, mult):
					result.append(newPhrase)

		if (len(result) > 200):
			result = result[:len(result) / 4]

		return ' '.join(result)

	def removeDuplicates(self, seq):
		seen = set()
		seen_add = seen.add
		return [x for x in seq if not (x in seen or seen_add(x))]

	def tagNPFilter(self, sentence):
		tokens = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(tokens)
		# NPgrammar = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
		# ND:{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""
		NPgrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
		#Problem: "a powerful computer with strong support from university" 
		#1, nested; 2, 'computer' is the keywords? or 'computer with support' is the keywords?
		cp = nltk.RegexpParser(NPgrammar)
		resultTree = cp.parse(tagged)   #result is of type nltk.tree.Tree
		result = ""
		stemmer = SnowballStemmer("english")
		for node in resultTree:
			if (type(node) == nltk.tree.Tree):
				#result += ''.join(item[0] for item in node.leaves()) #connect every words
				#result += stemmer.stem(node.leaves()[len(node.leaves()) - 1][0]) #use just the last NN

				if node[0][1] == 'DT':
					node.remove(node[0])  #remove the determiners
				currNounPhrase = ''.join(stemmer.stem(item[0]) for item in node.leaves())
				result += currNounPhrase

				if len(node.leaves()) == 1:
					pass
				else:
					result += ' '
					result += currNounPhrase #double noun phrases to increase the weight

				### The following part assumes nested grammar can be supported ###
				### which turns out to be false, so use the previous selction instead ###
				# if (node.label() == 'NP'):   # NN phrases
				# 	result += node.leaves()[len(node.leaves()) - 1][0]
				# else:    # IN phrases
				# 	if (node[0][1] == 'NN' or node[0][1] == 'NNS'):    # the first element is NN
				# 		result += node[0][0]
				# 	else:    # the first element is DT
				# 		result += node[1][0]
				### End of wasted part ###

			else:
				result += stemmer.stem(node[0])
			result += " "
		return result

	def keyWordFilter(self, article, keywords, filterTags):
		# the idea is to use citation, reference, keyword list to find software engineering related articles
		for filterTag in filterTags:
			tagContents = article.getElementsByTagName(filterTag)

			for tagContent in tagContents:
				for keyword in keywords:
					keywordDash = '-'.join(keyword.split(' '))
					if ((keyword in tagContent.childNodes[0].data.lower()) 
						or (keywordDash in tagContent.childNodes[0].data.lower())):
						return True
		
		return False

	def labelAllocator(self, article):
		labels = ['agile software-development', 'program-comprehension program-visualization',
			'autonomic self-managed software', 'requirements engineering',
			'computer-supported collaborative work', 'reengineering reverse-engineering',
			'component-based software-engineering', 'quality performance',
			'configuration management deployment', 'service-oriented architectures applications',
			'dependendability safety reliability', 'software-architecture design',
			'distributed web-based internet-scale', 'software-economics software-metrics',
			'empirical software-engineering', 'software-evolution',
			'end-user software-engineering', 'software-maintenance',
			'engineering secure software', 'software-policy software-ethics',
			'feature interaction generative programming', 'software-reuse',
			'human social aspects', 'software-specifications',
			'knowledge-based software-engineering', 'testing analysis',
			'mobile embedded real-time systems', 'theory formal-methods',
			'model-driven software-engineering', 'tools environments',
			'patterns frameworks', 'validation verification',
			'processes workflow']
		labelCheckList = [0] * len(labels)
		returnedLabels = []
		targetTags = ['par', 'title', 'subtitle', 'ft_body', 'concept_desc', 'kw']
		contentString = ''
		for tag in targetTags:
			tagContents = article.getElementsByTagName(tag)
			for tagContent in tagContents:
				contentString += tagContent.childNodes[0].data.lower()
				contentString += ' '
		for i in range(0, len(labels)):
			label = labels[i]
			tokens = label.split(' ')
			for token in tokens:
				origin = token
				spaceDuplicate = token.replace('-', ' ')
				if origin in contentString or spaceDuplicate in contentString:
					labelCheckList[i] += 1

			if labelCheckList[i] >= len(tokens): # Q: how to set this threshold
				returnedLabels.append(','.join(tokens))

		if len(returnedLabels) > 0:
			return ' '.join(returnedLabels)
		else:
			return 'none'

	def parse(self):

		count = 1
		result = ""

		keywords = ["software engineering", "software and its engineering"]
		filterList = ["kw", "ref_text", "cited_by_text", "concept_desc", "subtitle"]

		for fileName in self.fileList:
			try:
				DOMTree = xml.dom.minidom.parse(fileName)
			except xml.parsers.expat.ExpatError, e:
				print "The file causing the error is: ", fileName
				print "The detailed error is: %s" %e
			collection = DOMTree.documentElement

			#articles = collection.getElementsByTagName("article_rec")
			articles = collection.getElementsByTagName(self.rootTag)
			regexBracket = re.compile(r'<.*?>', re.IGNORECASE)
			regexQuote = re.compile(r'\"', re.IGNORECASE)

			for article in articles:

				if not self.keyWordFilter(article, keywords, filterList):
					pass
				else:
					# add in label filters to allocate the labels
					# tags = self.labelAllocator(article)
					for tag in self.targetTags:
						tagContents = article.getElementsByTagName(tag)

						if (tag != "article_publication_date"):
							for tagContent in tagContents:
								tagContent = re.sub(r'<.*?>', "", tagContent.childNodes[0].data)
								tagContent = re.sub(r'\"', "", tagContent)
								tagContent = str(tagContent.encode('utf-8')).translate(None, string.punctuation)
								#tagContent = str(tagContent).translate(None, string.punctuation)
								tagContent = ''.join([i for i in tagContent if not i.isdigit()])
								# tagContent = regexBracket.sub("", tagContent)
								# tagContent = regexQuote.sub("", tagContent)
								# tagContent = self.tagNPFilter(tagContent)

								### this is the part of getting keyphrases ###
								tagContent = self.getKeyphraseByTextRank(tagContent)
								self.content += tagContent
								self.content += " "

						else:
							for time in tagContents:
								timeList = time.childNodes[0].data.split("-")
								timing = timeList[len(timeList) - 1]
								self.content = timing + " " + self.content
								break
					# titles = article.getElementsByTagName("title")
					# abstracts = article.getElementsByTagName("par")
					# timeStamp = article.getElementsByTagName("article_publication_date")

					# for title in titles:
					# 	title = regexBracket.sub("", title)
					# 	title = regexQuote.sub("", title)
					# 	self.content += title
					# 	self.content += " "

					# for abstract in abstracts:
					# 	abstract = regexBracket.sub("", abstract)
					# 	abstract = regexQuote.sub("", abstract)
					# 	self.content += abstract
					# 	self.content += " "

					# for time in timeStamp:
					# 	timeList = time.split('-')
					# 	timing = timeList[len(timeList) - 1]
					# 	self.content = timing + " " + self.content
					if (len(self.content.split(" ")) <= 20):
						continue

					if self.content:
						# temp = [str(self.content.strip().encode('utf-8'))]
						# print temp
						# self.content = self.getKeyphraseByTextRank(temp)
						result += (str(count) + " en " + self.content + "\n")
						# result += (str(count) + "\t" + tags + "\t" + self.content + "\n")
						self.content = ""
						count += 1

		return result