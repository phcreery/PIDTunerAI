import miniEvolve as Evolution
#import math
import lib.tools as tools
import random
from terminalplot import plot
#import asciiplotlib as apl
#import matplotlib.pyplot as plt
import numpy as np
#import csv
#import tflearn
#from tflearn.layers.core import input_data, fully_connected
#from tflearn.layers.estimator import regression
import time

import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

#import pandas as pd


debug = False

profile = [
	[0,0],
	[1,0],
	[1.1,10],
	[5,10],
	]
profile = tools.datariser(profile, "Profile")
profile.fill(0.1)


t,sp,op,pv = tools.sampledata()

spdata = tools.mesh(t,sp)
opdata = tools.mesh(t,op)
pvdata = tools.mesh(t,pv)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PID:
	def __init__(self, kp=1, ki=1, kd=1):
		self.ki = ki
		self.kp = kp
		self.kd = kd
		self.lastNow = 0 #datetime.datetime.now()
		self.iterm = 0
		self.lastErr = 0

	def compute(self, setpoint, ispoint, now):
		#now = datetime.datetime.now()
		timeDelta = (now - self.lastNow)#.total_seconds()

		error = float(setpoint - ispoint)
		self.iterm += (error * timeDelta * self.ki)
		self.iterm = sorted([-1, self.iterm, 1])[1]
		dErr = (error - self.lastErr) / timeDelta

		output = self.kp * error + self.iterm + self.kd * dErr
		output = sorted([-1, output, 1])[1]
		self.lastErr = error
		self.lastNow = now

		return output


class Simulator:
	def __init__(self, name, calculator):
		self.name = name
		self.calculator = calculator

	def generate(self, PIDs):
		p,i,d = PIDs
		calc = self.calculator(p,i,d)
		y = 0
		v = 0
		dt = self.SP[1][0] - self.SP[0][0]
		OP = []
		PV = []
		for element in self.SP:
			t = element[0]
			force = calc.compute(element[1],y,element[0])
			PV.append([t,force])
			#F=ma a=dv*dt
			#dv = F/m*dt
			dv = (force/self.mass)*dt
			v = v + dv
			v = self.clamp(v, -self.maxVel, self.maxVel)
			y = y + v/dt + random.uniform(-self.noise,self.noise)
			if t < self.delay:
				y = 0
			OP.append([t,y])
			
			#print("x:",t,"Y:",y,"Force;", force,dt, dv, v)
		return OP, PV

	def clamp(self, v, low, high):
		if v > high:
			v = high
		if v < low:
			v = low
		return v

	def setpoint(self, SP):
		self.SP = SP

	def randomize(self):
		self.mass = random.uniform(0,300)
		self.maxVel = random.uniform(0.01,5)
		self.delay = random.uniform(0,10)
		self.noise = random.uniform(0,0.01)
		#self.weight = 0

	def randomPID(self):
		self.p = random.uniform(0,10)
		self.i = random.uniform(0,10)
		self.d = random.uniform(0,10)
		return [self.p,self.i,self.d]
		
	def show(self):
		print("m:", self.mass, "mV:", self.maxVel, "d:", self.delay, "N:", self.noise)

	def setScorer(self, calculator):
		self.calculator = calculator

	def generateAndScore(self, PIDs):
		output, PV = self.generate(PIDs)
		score = -tools.weightedAreaBetween(self.SP,output)
		return score




class Database:
	def __init__(self, filename="traindata.npy"):
		self.filename = filename
		self.f = open(self.filename,"a+")# as csvfile
		#header = "x\ny"
		#np.savetxt(self.f, [], header=header)
		#self.csvwriter = csv.writer(self.f)
		#self.csvwriter.writerow(["x","y"])

	def create(self, amount):
		print()

	def add(self, x, y):
		#data = str(x)+str(y)+"\n"
		#print(data)
		#self.f.write(data)

		X1 = np.array(x[0])
		X2 = np.array(x[1])
		Y = np.array(y)
		Z = np.concatenate([X1,X2,Y])
		#print(Z)
		np.savetxt(self.f, [Z], delimiter=",")

	def read(self):
		#self.f2 = open(self.filename,"r")
		#new_data = np.loadtxt(self.filename)
		data = np.loadtxt(self.filename, delimiter=",")
		return data

	def finish(self):
		self.f.close() 



def generateData(count = 1, databasename = "traindata-new.npy"):
	global PID#, Simulator

	sim = Simulator("sim", PID)
	sim.setpoint(spdata)
	pop = Evolution.Population('colony', 50, 25)
	pop.setFitnessCalculator(sim.generateAndScore)
	pop.setResultRange(0,30)
	flashcards = Database(databasename)
		
	counter = 0
	while counter < count:
		print()
		print("Training Data #", counter, bcolors.OKBLUE)
		start = time.time()

		# Create Real-World Scenario and data with Basic Profile
		sim.randomize()

		randomPID = sim.randomPID()
		#print(randomPID)
		pop.Initialize(randomPID, 6)
		pop.Evolve(1, "Scenario")		   #make the input seem reasonable
		scenarioPID = pop.getTopScore()
		OP, PV = sim.generate(scenarioPID)
		opdata = tools.datariser(OP)
		if debug == True: plot(opdata.getx(),opdata.gety())
		#print("Area: ",tools.areaBetween(spdata,opdata.getdata()))


		# Find answer to scenario
		pop.Initialize(list(scenarioPID), 6)
		
		pop.Evolve(60, "Solution")
		correctPID = pop.getTopScore()
		p,i,d = correctPID
		i=i/2
		OP, PV = sim.generate((p,i,d)) #sim.generate(pop.getTopScore())
		newopdata = tools.datariser(OP)
		newpvdata = tools.datariser(PV)
		#print(opdata.getx())
		if debug == True: plot(newopdata.getx(),newopdata.gety())
		if debug == True: sim.show()
		if debug == True: tools.saveoutput([[newopdata.getx(),newopdata.gety()],[t,sp], [newpvdata.getx(),newpvdata.gety()]], "lastGenerated")

		flashcards.add( [scenarioPID,opdata.gety()], correctPID )
		counter = counter + 1



		end = time.time()
		time_taken = end - start
		#print('Time: ', )
		print(bcolors.ENDC, " Time:", round(time_taken), "Final Score:", round(-tools.weightedAreaBetween(spdata,newopdata.getdata()), 1))


	flashcards.finish()




class AI:
	def __init__(self, epoch = 3, initial_data = 1000, database = "traindata.npy", test_times = 1, lr = 1e-2, modelname = 'modelv1.model'):
		#self.initial_data = initial_data
		#self.initial_data_filename = initial_data_filename
		self.test_times = test_times
		self.lr = lr	#learning rate?
		self.modelname = modelname
		self.epoch = epoch
		self.trainingdb = Database(database)

	def initial_population(self):
		generateData(self.initial_data)

	def load_trainingdata(self):
		print("Loading Training Data...")
		training_data = self.trainingdb.read()
		#print(training_data)

		#X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
		#y = np.array([i[1] for i in training_data]).reshape(-1, 1)
		
		#for row in training_data:
		#	data = np.split(row, [-3])

		X = np.array([np.split(row, [-3])[0] for row in training_data])#.reshape(-1, 314, 1, 1)
		y = np.array([np.split(row, [-3])[1] for row in training_data])

		#X = X[:, :, :, np.newaxis]
		#y = np.reshape(y.shape[0], -1)

		X = tf.keras.utils.normalize(X, axis=0)
		y = tf.keras.utils.normalize(y, axis=1)

		
		print(X, end="\n\n")
		print(y, end="\n\n")
		print("X:", X.size, X.shape)
		print("Y:", y.size, y.shape)
		
		#exit()

		return X,y

	def CNNmodel(self, inputsize):
		model = tf.keras.models.Sequential()
		#model.add(tf.keras.layers.Flatten())

		"""
		Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
		"""

		model.add(tf.keras.layers.Conv2D(64, (1, 3), activation='relu', input_shape=[1, 314, 1]))
		#model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))

		#model.add(tf.keras.layers.Conv2D(64, (1, 3)))
		#model.add(tf.keras.layers.Activation('relu'))
		#model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))

		model.add(tf.keras.layers.Flatten())

		#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[314]))
		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
		#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
		model.add(tf.keras.layers.Dense(3))
		#optimizer = tf.keras.optimizers.RMSprop(0.001)		
		model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # categorical_crossentropy,  binary_crossentropy

		return model


	def model(self, inputsize):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[1, 314]))
		model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
		#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
		model.add(tf.keras.layers.Dense(3))
		model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])  # categorical_crossentropy, binary_crossentropy

		return model

	def train_model(self, training_data, model):
		#early_stop = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=10)
		
		X,y = training_data
		self.history = model.fit(X, y, epochs = self.epoch)  # callbacks=[early_stop]
		#model.fit(X,y, n_epoch = self.epoch, shuffle = True, run_id = self.filename)
		model.save(self.modelname)
		return model

	def train(self):
		training_data = self.load_trainingdata()
		#print("Input Size:", training_data[0].shape[1:], training_data[1].shape)
		#exit()
		nn_model = self.model(training_data[0].shape[1:])
		self.nn_model = self.train_model(training_data, nn_model)
		#self.test_model(nn_model)

	def predict(self, inputdata, modelname = None):
		if modelname == None: modelname = self.modelname
		model = tf.keras.models.load_model(modelname)
		inputdata = np.array(inputdata).reshape(-1, 314)
		#inputdata.reshape(314, 1)
		#print(inputdata)
		predictions = model.predict(inputdata)
		#print(predictions)
		return predictions

	def plot_history(self):
		history = self.history
		#hist = pd.DataFrame(history.history)
		#hist['epoch'] = history.epoch
		#hist.tail()
		print(history.history.keys())

		# Get training and test loss histories
		loss = history.history['loss']
		acc = history.history['acc']

		# Create count of the number of epochs
		epoch_count = range(1, len(loss) + 1)

		tools.saveoutput([[epoch_count, acc], [epoch_count, loss]], "train-hist")


if __name__ == "__main__":
	#generateData(2, "traindata.npy")
	ai = AI(epoch = 60)
	ai.train()
	ai.plot_history()
	
	#generateData(1, "testdata.npy")
	testdb = Database("testdata.npy")
	testdata = testdb.read()
	testdb.finish()
	X = np.array([np.split(row, [-3])[0] for row in testdata])
	y = np.array([np.split(row, [-3])[1] for row in testdata])
	predictions = ai.predict(X, modelname = "modelv1.model")
	print("Actual:\n", y)
	print("Predictions:\n", predictions)
	

