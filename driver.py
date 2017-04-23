import os
import csv
from sklearn.linear_model import SGDClassifier



train_path = "../aclImdb/train/" # Change your path accordingly.
test_path = "../imdb_te.csv" # test data for grade evaluation. Change your path accordingly.

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
	'''Implement this module to extract
	and combine text files under train_path directory into
	imdb_tr.csv. Each text file in train_path should be stored 
	as a row in imdb_tr.csv. And imdb_tr.csv should have three 
	columns, "row_number", "text" and label'''


	#Collect all the english stop words in a list for later comparison
	englishstopWords = []
	with open(inpath+"stopwords.en.txt", "r") as stopwordsFile:
		for line in stopwordsFile:
			englishstopWords.append(line.rstrip('\n'));

			#print(englishstopWords)


	#List of each file in the positive/negative example folders
	posFolder = os.listdir(train_path+"pos/")
	negFolder = os.listdir(train_path+"neg/")


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
			
			positiveFile = open(train_path + "pos/" + positiveFileName)

			#Store file as a single string
			positiveText = positiveFile.read()

			#We dont need the file after storing it in a string
			positiveFile.close()

			#Create a list of words found in the single string
			words = positiveText.split()#('; |, |. |\n')#(' ',',', '.', ';', '!','?','\"',"-", "(",")")# or also whitesoace and punctuation

			#Remove all words found in the english stop words list
			for word in words:
				if word in englishstopWords or word.lower() in englishstopWords:
					words.remove(word)

			#Recreate the single text string
			text = ' '.join(word.lower() for word in words)

			writer.writerow({'row_number': str(count), 'text':text, 'polarity': '1'})
			count += 1

		
		count = 0
		#Iterate through each negative file and write it to a row in main outfile	
		for negativeFileName in posFolder:
			
			negativeFile = open(train_path + "pos/" + negativeFileName)

			#Store file as a single string
			negativeText = negativeFile.read()

			#We dont need the file after storing it in a string
			negativeFile.close()

			#Create a list of words found in the single string
			words = negativeText.split()#(' ',',', '.', ';', '!','?','\"',"-", "(",")")# or also whitesoace and punctuation

			#Remove all words found in the english stop words list
			for word in words:
				if word in englishstopWords or word.lower() in englishstopWords:
					words.remove(word)

			#Recreate the single text string
			text = ' '.join(word.lower() for word in words)

			writer.writerow({'row_number': str(count), 'text':text, 'polarity': '0'})
			count += 1






























	# stopwordsFile.close()
	# outfile.close()








if __name__ == "__main__":
	'''train a SGD classifier using unigram representation,
	predict sentiments on imdb_te.csv, and write output to
	unigram.output.txt'''
		
	'''train a SGD classifier using bigram representation,
	predict sentiments on imdb_te.csv, and write output to
	unigram.output.txt'''
	 
	'''train a SGD classifier using unigram representation
	with tf-idf, predict sentiments on imdb_te.csv, and write 
	output to unigram.output.txt'''
		
	'''train a SGD classifier using bigram representation
	with tf-idf, predict sentiments on imdb_te.csv, and write 
	output to unigram.output.txt'''
	


	imdb_data_preprocess("")