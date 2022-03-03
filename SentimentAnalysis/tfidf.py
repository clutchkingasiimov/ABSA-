from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from LCSAlgo import LCSSetAlgorithm
from nltk.stem import WordNetLemmatizer
import nltk 
nltk.download('punkt')
nltk.download('stopwords')


class Plagiarism():

	stopwords = nltk.corpus.stopwords.words('english')

	def __init__(self, text1, text2):
		self.text1 = text1
		self.text2 = text2
		self.texts = [self.text1, self.text2]

	#Creates tokens of the sentence (Usage is for jaccard similarity or raw Tf-Idf checks)
	def __text_tokens(self, str_to_tokenize):
		tokens = nltk.word_tokenize(str_to_tokenize) #Tokenizes the sentence
		tokens = [token.strip().lower() for token in tokens] #Stores them in a list and makes them all lowercase
		return tokens

	#Removes all the stopwords (Usage is optional)
	def __stopwords_remover(self, str_to_tokenize):
		sentence_tokens = self.__text_tokens(str_to_tokenize) #Tokenize the sentence
		token_filters = [token for token in sentence_tokens if token not in self.stopwords] #Keep all the tokens that are not stopwords
		new_text = ' '.join(token_filters) #Join the processed sentence back
		return new_text

	#Removes special characters like ".", "!", "$", etc
	def __remove_characters(self, str_to_tokenize):
		sentence_tokens = self.__text_tokens(str_to_tokenize) #First we create the tokens of the sentence
		special_character_vals = re.compile('[{}]'.format(re.escape(string.punctuation))) #Keeping the special characters, we re-format the punctuation
		token_filters = filter(None, [special_character_vals.sub(' ', token) for token in sentence_tokens])
		clean_text = ' '.join(token_filters)
		return clean_text

	#Creates a bag of words where each word is tracked by its count
	# def bag_of_words(self, clean_text1, clean_text2):
	# 	word_dict = gensim.corpora.Dictionary(clean_text1) #First store the words in a corpora dictionary
	# 	word_corpus = [] #Create the word-corpus
	# 	sim_check = [] #Create a similarity check list
	# 	#With a for-loop, count the frequency in which each word appears in the sentences and store them in "word_corpus"
	# 	for text in clean_text1:
	# 		word_corpus.append(word_dict.doc2bow(text))
	#

	def tf_idf_transformation(self, clean_text1, clean_text2):
		tfidf = TfidfVectorizer()
		X = tfidf.fit_transform([clean_text1, clean_text2])
		pairwise_similarity = X * X.T

		return ((pairwise_similarity).A)[0,1]


	def compute_pairwise_similarity(self):
		processed = self.__remove_characters(self.text1)
		processed_2 = self.__remove_characters(self.text2)
		pairwise_metric = self.tf_idf_transformation(processed, processed_2)

		lcs_algo = LCSSetAlgorithm(processed, processed_2)
		lcs_pairwise_similarity = lcs_algo.normalized_lcs()

		pairwise_score = 100 * ((0.5*pairwise_metric) + (0.5*lcs_pairwise_similarity))


		print("Sentence 1: {}".format(self.text1))
		print("Sentence 2: {}".format(self.text2))
		print("Text Similarity Detected: {}%".format(np.round(pairwise_score)))






######################################################3
sentence = "Python is an important programming language"
sentence_2 = "Learning to program in python is important    "
detection = Plagiarism(sentence, sentence_2)
sim_check = detection.compute_pairwise_similarity()
# print(sim_check)