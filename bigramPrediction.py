import os
import csv
import pandas
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import numpy
from sklearn.model_selection import train_test_split



def predictWithBigram(infile_name):
	"""Function to build a more sophisticated data representation model; namely,
	the bigram model. This is where occurrences depend on a sequence
	 of two words rather than an individual one.
	 Then trains Stochastic Descend Gradient (SGD) classifier."""

	print("Start prediction with bigram.")

	textRows = []
	labels = []

	#Train the classifier
	with open(infile_name, "r", encoding = "utf-8") as csv_infile:
		
		csv_reader = csv.DictReader(csv_infile)

		#Convert each row string into unigram representation & update vocab set
		for row in csv_reader:

			rowText = row["text"]
			textRows.append(rowText)

			label = row["polarity"]
			labels.append(label)


	countVectorizer = CountVectorizer(stop_words = None, encoding = 'utf-8', ngram_range = (2, 2))

	fit_transform = countVectorizer.fit_transform(textRows)
	vocab = countVectorizer.vocabulary_

	labels = numpy.asarray(labels)

	SGDclassifier = SGDClassifier(loss = 'hinge', penalty = 'l1')
	SGDclassifier.fit(fit_transform, labels)

	countVectorizer = CountVectorizer(vocabulary = vocab, ngram_range = (2, 2))


	outfile = open('bigram.output.txt','w')

	textArray = []
	with open('imdb_te.csv','r',encoding = 'ISO-8859-1') as csvFile:


		csvReader = csv.DictReader(csvFile)
		time1_0 = time.clock()
		for fileRow in csvReader:

			text = fileRow['text']

			textArray.append(text)


	bigram = countVectorizer.fit_transform(textArray)
	predictions = SGDclassifier.predict(bigram)


	for prediction in predictions:
		outfile.write(str(prediction) + '\n')
	outfile.close()
	print("End prediction with bigram.")


	#Lastly, perform cross validation on the training set to determine accuracy
	x_train, x_test, y_train, y_test = train_test_split(fit_transform, labels, test_size=0.2)
	SGDclassifier = SGDClassifier(loss = 'hinge', penalty = 'l1')
	SGDclassifier.fit(x_train, y_train)
	print("Score bigram!")
	print(SGDclassifier.score(x_test, y_test))











def predictWithBigramTFIDF(infile_name):
	"""Function to transform text columnsin imdb_tr.csv into term-document
	 matrices using unigram model. 
	 Then trains Stochastic Descend Gradient (SGD) classifier."""

	print("Start prediction with bigram TFIDF.")

	textRows = []
	labels = []

	#Train the classifier
	with open(infile_name, "r", encoding = "utf-8") as csv_infile:
		
		csv_reader = csv.DictReader(csv_infile)

		#Convert each row string into unigram representation & update vocab set
		for row in csv_reader:

			rowText = row["text"]
			textRows.append(rowText)

			label = row["polarity"]
			labels.append(label)


	tfidfVectorizer = TfidfVectorizer(stop_words = None, encoding = 'utf-8', ngram_range = (2, 2))

	fit_transform = tfidfVectorizer.fit_transform(textRows)
	vocab = tfidfVectorizer.vocabulary_

	labels = numpy.asarray(labels)

	SGDclassifier = SGDClassifier(loss = 'hinge', penalty = 'l1')
	SGDclassifier.fit(fit_transform, labels)

	tfidfVectorizer = TfidfVectorizer(vocabulary = vocab, ngram_range = (2, 2))


	outfile = open('bigramtfidf.output.txt','w')

	textArray = []
	with open('imdb_te.csv','r',encoding = 'ISO-8859-1') as csvFile:


		csvReader = csv.DictReader(csvFile)
		for fileRow in csvReader:

			text = fileRow['text']

			textArray.append(text)


	bigram = tfidfVectorizer.fit_transform(textArray)
	predictions = SGDclassifier.predict(bigram)


	for prediction in predictions:
		outfile.write(str(prediction) + '\n')
	outfile.close()
	print("End prediction with bigramtfidf.")


	#Lastly, perform cross validation on the training set to determine accuracy
	x_train, x_test, y_train, y_test = train_test_split(fit_transform, labels, test_size=0.2)
	SGDclassifier = SGDClassifier(loss = 'hinge', penalty = 'l1')
	SGDclassifier.fit(x_train, y_train)
	print("Score!")
	print(SGDclassifier.score(x_test, y_test))
