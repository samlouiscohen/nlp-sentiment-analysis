import os
import csv
import pandas
from sklearn.linear_model import SGDClassifier
import re
from unigramPrediction import predictWithUnigram, predictWithUnigramTFIDF
from bigramPrediction import predictWithBigram, predictWithBigramTFIDF
import time



train_path = "../aclImdb/train/" # Change your path accordingly.
test_path = "../imdb_te.csv" # test data for grade evaluation. Change your path accordingly.

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
	'''Implement this module to extract
	and combine text files under train_path directory into
	imdb_tr.csv. Each text file in train_path should be stored 
	as a row in imdb_tr.csv. And imdb_tr.csv should have three 
	columns, "row_number", "text" and label'''

	print("Start Preprocessing.")

	#Collect english stop words to later filter out of natural text data ('the', 'is', 'at', ...)
	englishstopWords = []
	with open("stopwords.en.txt", "r") as stopwordsFile:
		for line in stopwordsFile:
			englishstopWords.append(line.replace('\n',''))#(line.rstrip('\n'))
	print(len(englishstopWords))

	#List of each file in the positive/negative example folders
	posFolder = os.listdir(inpath+"pos/")
	negFolder = os.listdir(inpath+"neg/")

	#Open the file to write our combined text files to
	with open(outpath+name, "w", encoding = "utf-8") as outfile:

		fieldNames = ["row_number", "text", "polarity"]

		#Create a writer object to write in CSV format (fieldNames as columns) to the outfile
		writer = csv.DictWriter(outfile, fieldnames = fieldNames, lineterminator = "\n")
		writer.writeheader()

		#Count tracks which number example is being written
		count = 0

		#Iterate through each positive file and write it to a row in main outfile	
		for positiveFileName in posFolder:
			
			positiveFile = open(inpath + "pos/" + positiveFileName)

			#Store file as a single string
			positiveText = positiveFile.read()

			#We dont need the file after storing it in a string
			positiveFile.close()

			#Create words from the text string, split on any non-alphabet characters (^ is "not")
			words = re.split('[^a-zA-z]', positiveText)
			#words.remove('') #this was included for some strange reason/then got error

			#Remove all words found in the english stop words list
			for word in words:

				if word in englishstopWords or word.lower() in englishstopWords:
					words.remove(word)

			#Recreate the single text string after removing excess punctuation and english stop words
			text = ' '.join(word.lower() for word in words)

			#Write each new cleaned string of words out to file
			writer.writerow({'row_number': str(count), 'text':text, 'polarity': '1'})
			count += 1

		#Iterate through each negative file and write it to a row in main outfile	
		for negativeFileName in negFolder:
			
			negativeFile = open(inpath + "neg/" + negativeFileName)

			#Store file as a single string
			negativeText = negativeFile.read()

			#We dont need the file after storing it in a string
			negativeFile.close()

			#Create words from the text string, split on any non-alphabet characters (^ is "not")
			words = re.split('[^a-zA-z]', negativeText)

			#Remove all words found in the english stop words list
			for word in words:

				if word in englishstopWords or word.lower() in englishstopWords:
					words.remove(word)

			#Recreate the single text string
			text = ' '.join(word.lower() for word in words)
			
			#Write each new cleaned string of words out to file
			writer.writerow({'row_number': str(count), 'text':text, 'polarity': '0'})
			count += 1


	print("Finished Preprocessing.")







if __name__ == "__main__":
	
	startTotal = time.clock()

	#Preprocess the training data
	start1 = time.clock()
	imdb_data_preprocess(inpath = "../aclImdb/train/", name="imdb_tr.csv")
	end1 = time.clock()
	print("Preprocessed training data and wrote out in: "+ str(end1-start1))

	#Preprocess the testing data
	start2 = time.clock()
	imdb_data_preprocess(inpath = "../", name = "imdb_te.csv")
	end2 = time.clock()
	print("Preprocessed testing data and wrote out in: "+ str(end2-start2))




	'''train a SGD classifier using unigram representation,
	predict sentiments on imdb_te.csv, and write output to
	unigram.output.txt'''
	start1 = time.clock()
	predictWithUnigram("imdb_tr.csv")
	end1 = time.clock()
	print("Stage one completed in: "+ str(end1-start1))
	#print("done stage 1.")
	#input()
		
	'''train a SGD classifier using bigram representation,
	predict sentiments on imdb_te.csv, and write output to
	unigram.output.txt'''
	start2 = time.clock()
	predictWithBigram("imdb_tr.csv")
	end2 = time.clock()
	print("Bigram completed in: "+ str(end2-start2))

	#input()
	 
	'''train a SGD classifier using unigram representation
	with tf-idf, predict sentiments on imdb_te.csv, and write 
	output to unigram.output.txt'''
	start3 = time.clock()
	predictWithUnigramTFIDF("imdb_tr.csv")
	end3 = time.clock()
	print("unigramTFIDF completed in: "+ str(end3-start3))
	#print("done stage 3.")
		
	'''train a SGD classifier using bigram representation
	with tf-idf, predict sentiments on imdb_te.csv, and write 
	output to unigram.output.txt'''
	start4 = time.clock()
	predictWithBigramTFIDF("imdb_tr.csv")
	end4 = time.clock()
	print("bigramTFIDF completed in: "+ str(end3-start3))
	
	endTotal = time.clock()
	print("Total running time: "+ str(endTotal-startTotal))

	
	



	
	

	









