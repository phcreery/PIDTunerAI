import random
import math
#from tqdm import tqdm
import sys

class Population:

	#creates array with values and score
	#  [[1,2,3][2]
	#	[2,3,4][4]
	#	...		  ]


	def __init__(self, name, size = 100, killcount = 50):
		self.name = name
		self.size = size
		self.killcount = killcount
		self.clampresults = False

	def showinfo(self):
		print('Label: ', self.name)
		print('Size:  ', len(self.Values))
		print('Population:  ', self.Values)

	def Initialize(self, vals, variance, resolution = 0.1):
		#newvals = [(val + round(random.uniform(val-variance, val+variance))) for val in vals]
		self.Variance = variance
		self.Values = [[[random.uniform(val-variance, val+variance) for val in vals],0] for x in range(self.size)]
		
		#self.Values = vals

	def Offspring(self, vals):
		l=len(vals)
		for x in range(self.size - l):
			#print("Fill", x)
			parent = self.Values[random.randrange(l)][0]
			#print(parent)
			#print(l)
			child = [random.uniform(val-self.Variance, val+self.Variance) for val in parent]

			if self.clampresults == True:
				low,high = self.resultrange
				child = [self.clamp(val, low, high) for val in child]

			childfitness = 0
			self.Values.append([child,childfitness])

	def setResultRange(self, low, high):
		self.clampresults = True
		self.resultrange = (low, high)

	def clamp(self, v, low, high):
		if v > high:
			v = high
		if v < low:
			v = low
		return v

	def setFitnessCalculator(self, func):
		self.GetScore = func

	def FitnessTest(self, vals):
		for val in vals:
			val[1] = self.GetScore(val[0])

	def SortValues(self, vals):
		vals.sort(key=lambda x: x[1], reverse=True)
		return vals


	def oneEvolution(self):
		print("Performing One Evolution...")

		#Score
		self.FitnessTest(self.Values)
		#Sort
		self.Values = self.sortValues(self.Values)
		#Kill
		self.Values = self.Values[:-self.killcount]
		#Populate
		self.Offspring(self.Values)

	def Evolve(self, evolutions, title = ""):
		topscore = 0
		evolution = 0
		print("Performing "+str(evolutions)+" Evolution...")
		while evolution < evolutions:
		#while topscore < stopscore:
		#for evolution in tqdm(range(evolutions)):
			#Score
			self.FitnessTest(self.Values)
			#Sort
			self.Values = self.SortValues(self.Values)
			topscore = self.Values[0][1]
			#Kill
			self.Values = self.Values[:-self.killcount]
			#Populate
			self.Offspring(self.Values)

			evolution = evolution + 1
			#print("\r\rTop Score:   ", topscore, "  Evolution:", evolution)
			progressBar(evolution, evolutions, "Evolving: "+title, str(round(topscore)))
			
			#print(self.Values[0])
		#print(self.Values[0][0])
		print()
		return self.Values[0][0]

	def getTopScore(self):
		return self.Values[0][0]

def progressBar(value, endvalue, text = "", endtext = "", bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))

	sys.stdout.write("\r"+text+" [{0}] {1}%".format(arrow + spaces, int(round(percent * 100)))+" "+endtext+"  ")
	sys.stdout.flush()