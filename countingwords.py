import sys
import yaml
from np import NLP_Practice
config = yaml.safe_load(open('train_config.yml'))
TRAIN_CONFIG_2 = config['count_word_exp']

class CountingWords(NLP_Practice):
	def __init__ (self):
		self.cw = NLP_Practice()
		self.X_train = self.cw.vectorFitTransform(self.loadData())

		#Uncomment to print results
		# print self.cw.getFeatureNames()
		# print self.getShape()

	def loadData(self):
		B1 = self.cw.getFile(TRAIN_CONFIG_2['post1'])
		B2 = self.cw.getFile(TRAIN_CONFIG_2['post2'])
		B3 = self.cw.getFile(TRAIN_CONFIG_2['post3'])
		B4 = self.cw.getFile(TRAIN_CONFIG_2['post4'])
		B5 = self.cw.getFile(TRAIN_CONFIG_2['post5'])

		content = self.cw.readFile(B1),\
				  self.cw.readFile(B2),\
				  self.cw.readFile(B3),\
				  self.cw.readFile(B4),\
				  self.cw.readFile(B5)
		return content

	def getShape(self):
		self.num_samples, num_features = self.X_train.shape
		print self.num_samples
		return "#samples: %d, #features: %d" % (self.num_samples, num_features)

	def newPost(self, content):
		self.new_post = content
		self.content_vector = self.cw.vectorTransform([content])
		return self.content_vector




