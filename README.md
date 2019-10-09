# PIDTunerAI
 A python script utilizing Artificial Intelligence to fine-tune PID variables
 *This is a proof of concept and is no where near finished, please do not use this for your PID controller*

## How it works
This code performes two seperate actions - Learning & Fixing

### Learning
1. First, using my own evolutionary AI (miniEvolve), the script first generates a large number of randomly simulated process data which is stored in traindata.npy. The simulation uses simple physics to simulate a flow, acceleration, momentum, gravity, etc. Each row of data contains the randomly generated PID values, the simulated process data over time, and the fine-tuned evolved PID values.
2. Using Tensorflow, the data is then processed through a neural network to create a model, which is stored in model.tflearn.

### Fixing
1. The user then inputs their own PID values and process values over time, and the program returns PID values that are generated using the tensorflow model.
*The input data must be attempting at setpoint of 1, since that is that the model is trained to do*

 ## Commands
 
 ### Generation
 The following command can be used to generate 1000 of the physics based simulation data for a setpoint of 1 at time 0
 `generateData(1000, "traindata.npy")`

 ### Training
 The following commands can be used to create the test data and plot the outcome
 ```
 ai = AI(epoch = 40, initial_data = 1000, database = "traindata.npy", test_times = 1, lr = 1e-2, modelname = 'modelv1.model')
 ai.train()
 ai.plot_history()
 ```
 ### Using
 The following commands take input data and run it through the model to find batter PID values
 ```
 testdb = Database("testdata.npy")
 testdata = testdb.read()
 testdb.finish()
 X = np.array([np.split(row, [-3])[0] for row in testdata])
 y = np.array([np.split(row, [-3])[1] for row in testdata])
 predictions = ai.predict(X, modelname = "modelv1.model")
 print("Actual:\n", y)
 print("Predictions:\n", predictions)
 ```
 
 ## TODO
 - [ ] Better Neural Network
 - [ ] More flexible input data
 - [ ] More realistic training data
 - [ ] Better visualization and use simplificaiton
 
 
