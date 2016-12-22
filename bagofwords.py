import yaml
from np import NLP_Practice
config = yaml.safe_load(open('train_config.yml'))
TRAIN_CONFIG_1 =  config['bag_of_word_exp']

class BagOfWords(NLP_Practice):
	def __init__ (self):
		self.bow = NLP_Practice()
		X = self.bow.vectorFitTransform(self.loadData())
		
		#Uncomment to print results
		print self.bow.getFeatureNames()
		print X.toarray().transpose()

	def loadData(self):
		A1 =  self.bow.getFile(TRAIN_CONFIG_1['content1'])
		A2 = self.bow.getFile(TRAIN_CONFIG_1['content2'])
		content = self.bow.readFile(A1),self.bow.readFile(A2)
		return content
