import nltk
from sklearn.feature_extraction.text import CountVectorizer

class NLP_Practice(CountVectorizer):
	"""Clustering - Finiding Related Posts"""
	
	def __init__ (self):
		"""Initialise sklearn vectorizer."""
		self.vectorizer = CountVectorizer(min_df=1)

	def getFile(self, file):
		"""Opens a file."""
		self.x = 10
		return open(file,'r')

	def readFile(self, file):
		"""Reads content in the file."""
		return file.read()

	def vectorFitTransform(self, data):
		"""We put the list of subject lines 
		into the fit_transform() function of 
		vectorizer, which does all the hard 
		vectorization work."""
		return self.vectorizer.fit_transform(data)

	def vectorTransform(self, data):
		"""We put the list of subject lines 
		into the fit_transform() function of 
		vectorizer, which does all the hard 
		vectorization work."""
		return self.vectorizer.transform(data)

	def getFeatureNames(self):
		"""Returns the words which vectorizer has 
		detected for count."""
		return self.vectorizer.get_feature_names()





		