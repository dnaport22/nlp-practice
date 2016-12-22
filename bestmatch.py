from countingwords import CountingWords
from np import NLP_Practice
import scipy as sp
import sys

class BestMatch(CountingWords, NLP_Practice):
	def __init__ (self):
		self.bm = CountingWords()
		self.bmNLP = NLP_Practice()
		self.bm.newPost("Imaging databases")

		self.best_doc = None
		self.best_distance = sys.maxint
		self.best_i = None

		# print self.bm.X_train
		print '='*30
		print self.bm.new_post
		print '='*30
		# print self.bm.X_train.getrow(1) - self.bm.content_vector

	def dist_raw(self,v1,v2):
		delta = v1-v2
		return sp.linalg.norm(delta.toarray()) #Squareroot((q(i)-p(i))**2)

	def findDistance(self):
		for i, post in enumerate(self.bm.loadData()):
			if post == self.bm.new_post:
				continue
			post_vector = self.bm.X_train.getrow(i)
			d = self.dist_raw(post_vector, self.bm.content_vector)
			print "=== Post %i with dist=%.2f: %s"%(i, d, post)
			if d < self.best_distance:
				self.best_distance = d
				self.best_i = i
		print "Best post is %i with dist=%.2f"%(self.best_i, self.best_distance)


Test = BestMatch()
Test.findDistance()

