def deriveData(data,spdata):
    inc = data[1][0] - data[0][0]
    slopemax = 0
    print("inc:",inc)
    for x in range(len(data)-1):
        #print(data[x])
        if (data[x][1] - data[x-1][1]) > slopemax:
            #print("Slopemax:", slopemax)
            slopemax = data[x][1] - data[x-1][1]
    print("Slopemax:", slopemax)
    slope = slopemax/inc
    
    delay=0
    for x in range(len(data)-1):
        if (data[x][1] - data[x-1][1]) > slopemax / 2:
            break
        if spdata[x][1] > 0:
            delay=delay+1
    print("Delay:",delay)

    noise = 0
    for x in range(1,len(data)-1):
        mdpt = (data[x-1][1] - data[x+1][1])/2 + data[x-1][1]
        noise = noise + abs(mdpt - data[x][1])
    noise = noise / (len(data)-2)
    print("Noise:",noise)

def generateData():
    maxvel = 10
    mass = 10
    #weight = 0
    noise = 0
    delay = 0
    print()

deriveData(pvdata,spdata)



class AI:
	def __init__(self, epoch = 3, initial_data = 1000, initial_data_filename = "traindata.npy", test_times = 1, lr = 1e-2, filename = 'model.tflearn'):
		#self.initial_data = initial_data
		self.initial_data_filename = initial_data_filename
		self.test_times = test_times
		#self.goal_steps = goal_steps
		self.lr = lr	#learning rate?
		self.filename = filename
		self.epoch = epoch

	def initial_population(self):
		generateData(self.initial_data)

	def model(self):
		network = input_data(shape=[None, 314, 1], name='input')
		network = fully_connected(network, 100, activation='relu')
		network = fully_connected(network, 100, activation='relu')
		network = fully_connected(network, 1, activation='linear')
		network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
		model = tflearn.DNN(network, tensorboard_dir='log')
		return model

	def train_model(self, training_data, model):
		#X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
		#y = np.array([i[1] for i in training_data]).reshape(-1, 1)
		
		#for row in training_data:
		#	data = np.split(row, [-3])
		X = np.array([np.split(row, [-3])[0] for row in training_data]).reshape(-1, 314, 1)
		y = np.array([np.split(row, [-3])[1] for row in training_data]).reshape(-1, 1)


		print(X)
		print(X.size, X.shape)
		print(y)
		#exit()

		model.fit(X,y, n_epoch = self.epoch, shuffle = True, run_id = self.filename)
		model.save(self.filename)
		return model

	def load_trainingdata(self):
		print("Loading Training Data...")
		trainingdata = Flashcards().read()
		print(trainingdata)
		return trainingdata

	def train(self):
		training_data = self.load_trainingdata()
		#exit()
		nn_model = self.model()
		nn_model = self.train_model(training_data, nn_model)
		#self.test_model(nn_model)

	def visualise(self):
		nn_model = self.model()
		nn_model.load(self.filename)
		#self.visualise_game(nn_model)

	def test(self):
		nn_model = self.model()
		nn_model.load(self.filename)
		self.test_model(nn_model)
