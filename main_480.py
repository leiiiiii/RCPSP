import time
import os
import glob
import numpy as np
import random
import re
from NN_Model_480 import createNeuralNetworkModel
from Env import runSimulation, runSimulation_input, activitySequence, activity
from openpyxl import Workbook
from openpyxl.styles import Border, Alignment, Side


t_start = time.time()

# user defined parameters
# problem parameters
timeDistribution = "deterministic"    # deterministic, exponential, uniform_1, uniform_2, ...

# CPU parameters
numberOfCpuProcessesToGenerateData = 4   # paoloPC has 16 cores
maxTasksPerChildToGenerateData = 4        # 4 is the best for paoloPC

# input state vector  parameters
numberOfActivitiesInStateVector = 6
rescaleFactorTime = 0.1
timeHorizon = 10

# random generation parameters
numberOfSimulationRunsToGenerateData = 1000
numberOfSimulationRunsToTestPolicy = 1

# train parameters
percentageOfFilesTest = 0.1
importExistingNeuralNetworkModel = False
numberOfEpochs = 3 #walk entire samples
learningRate = 0.001

# paths
relativePath = os.path.dirname(__file__)
# absolutePathProjects = relativePath + "/RG30_merged/"
absolutePathProjects = relativePath + "/J30/"
absolutePathExcelOutput = relativePath + "/database_480/tflearn_1000.xlsx"

# other parameters
np.set_printoptions(precision=4)    # print precision of numpy variables

# initialise variables
numberOfActivities = None
numberOfResources = None
activitySequences = []
decisions_indexActivity = []
decisions_indexActivityPowerset = []
states = []
actions = []

# read all activity sequences from database
absolutePathProjectsGlob = absolutePathProjects + "*.txt"
files = sorted(glob.glob(absolutePathProjectsGlob))

# divide all activity sequences in training and test set
numberOfFiles = len(files)
numberOfFilesTest = round(numberOfFiles * percentageOfFilesTest)
numberOfFilesTrain = numberOfFiles - numberOfFilesTest
indexFiles = list(range(0, numberOfFiles))
indexFilesTrain = []
indexFilesTest = []
for i in range(numberOfFilesTest):
    randomIndex = random.randrange(0, len(indexFiles))
    indexFilesTest.append(indexFiles[randomIndex])
    del indexFiles[randomIndex]#delete
indexFilesTrain = indexFiles

# organize the read activity sequences in classes
for i in range(numberOfFiles):
    file = files[i]
    # print(File)
    # create a new activitySequence object
    currentActivitySequence = activitySequence()
    with open(file,"r") as f:
        currentActivitySequence.index = i
        currentActivitySequence.fileName = os.path.basename(f.name)
        # print(currentActivitySequence.fileName)
        # allLines = f.read()
        # print(allLines)
        # next(f)
        firstLine = f.readline()    # information about numberOfActivities and numberOfResourceTypes
        firstLineDecomposed = re.split(" +", firstLine)
        numberOfActivities = (int(firstLineDecomposed[0])-2)    # the first and last dummy activity do not count
        currentActivitySequence.numberOfActivities = numberOfActivities
        # print("numberOfActivities = " + str(currentActivitySequence.numberOfActivities))
        secondLine = f.readline()   # information about total available resources
        secondLineDecomposed = re.split(" +", secondLine)
        numberOfResources = 0
        # print(len(secondLineDecomposed))
        # secondLineDecomposed=[int(secondLineDecomposed)]
        # print(secondLineDecomposed)
        for totalResources in secondLineDecomposed[0:-1]:
            numberOfResources += 1
            currentActivitySequence.totalResources.append(int(totalResources))
            # print(currentActivitySequence.totalResources)
        currentActivitySequence.numberOfResources = numberOfResources
        thirdLine = f.readline()   # information about starting activities
        thirdLineDecomposed = re.split(" +", thirdLine)
        for IdActivity in thirdLineDecomposed[6:-1]:
            currentActivitySequence.indexStartActivities.append(int(IdActivity)-2)
        #print("indexStartingActivities = " + str(currentActivitySequence.indexStartActivities))
        line = f.readline()
        while line:
            #print(line, end="")
            lineDecomposed = re.split(" +", line)
            if int(lineDecomposed[0]) == 0:
                break
            else:
                currentActivity = activity()
                currentActivity.time = int(lineDecomposed[0])
                currentActivity.requiredResources = [ int(lineDecomposed[1]),int(lineDecomposed[2]),int(lineDecomposed[3]),int(lineDecomposed[4]) ]
                for IdFollowingActivity in lineDecomposed[6:-1]:
                    if int(IdFollowingActivity) != numberOfActivities+2:    #if the following action is not the last dummy activity
                        currentActivity.indexFollowingActivities.append(int(IdFollowingActivity) - 2)
            currentActivitySequence.activities.append(currentActivity)
            line = f.readline()
        #add indexes to list of activities
        for j in range(len(currentActivitySequence.activities)):
            currentActivitySequence.activities[j].index = j
        #add numberOfPreviousActivities
        for Activity in currentActivitySequence.activities:
            for IndexFollowingActivity in Activity.indexFollowingActivities:
                currentActivitySequence.activities[IndexFollowingActivity].numberOfPreviousActivities += 1
    activitySequences.append(currentActivitySequence)

stateVectorLength = numberOfActivitiesInStateVector + numberOfActivitiesInStateVector * numberOfResources + numberOfResources

# compute decisions: each decision corresponds to a start of an activity in the local reference system (more than one decision can be taken at once)
for i in range(0,numberOfActivitiesInStateVector):
    decisions_indexActivity.append(i)

####  GENERATE TRAINING DATA USING RANDOM DECISIONS (WITHOUT USING pool.map) ####
for i in range(numberOfFilesTrain):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.indexActivitySequence = indexFilesTrain[i]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "generateData"
    currentRunSimulation_input.randomDecisionProbability = 1
    currentRunSimulation_input.policyType = None
    currentRunSimulation_input.decisionTool = None
    currentRunSimulation_input.activitySequences = activitySequences
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTrain[i]].totalDurationMean = currentRunSimulation_output.totalDurationMean
    activitySequences[indexFilesTrain[i]].totalDurationStandardDeviation = currentRunSimulation_output.totalDurationStDev
    activitySequences[indexFilesTrain[i]].totalDurationMin = currentRunSimulation_output.totalDurationMin
    activitySequences[indexFilesTrain[i]].totalDurationMax = currentRunSimulation_output.totalDurationMax
    activitySequences[indexFilesTrain[i]].luckFactorMean = currentRunSimulation_output.luckFactorMean
    activitySequences[indexFilesTrain[i]].trivialDecisionPercentageMean = currentRunSimulation_output.trivialDecisionPercentageMean
    for currentStateActionPair in currentRunSimulation_output.stateActionPairsOfBestRun:
        states.append(currentStateActionPair.state)
        actions.append(currentStateActionPair.action)
    #correspondence best states and actions pairs --> len(states) = len(actions)
    # print('states:',states)
    #print('length of states:',len(states))
    # print('actions:',actions)
    #print('length of actions:', len(actions))
    #print('##############################################################################################################################################')

    # print(states[0]) # the first element of states
    # print(actions[0])


####  TRAIN MODEL USING TRAINING DATA  ####
# look for existing model
if importExistingNeuralNetworkModel:
    neuralNetworkModelAlreadyExists = False
    print("check if a neural network model exists")
    if neuralNetworkModelAlreadyExists:
        print("import neural network model exists")
    else:
        neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
        # neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]))
else:
    neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
    # neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]))

# states_keras = np.reshape(states,(-1, len(states[0])))
# actions_keras = np.reshape(actions,(-1, len(actions[0])))
# history = LossHistory

neuralNetworkModel.fit({"input": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_step=500, show_metric=True)
# neuralNetworkModel.fit( states_keras, actions_keras, epochs=numberOfEpochs,callbacks=[history])

####  CREATE BENCHMARK WITH RANDOM DECISIONS ALSO WITH TEST ACTIVITY SEQUENCES  ####
for i in range(numberOfFilesTest):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.indexActivitySequence = indexFilesTest[i]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 1
    currentRunSimulation_input.policyType = None
    currentRunSimulation_input.decisionTool = None
    currentRunSimulation_input.activitySequences = activitySequences
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTest[i]].totalDurationMean = currentRunSimulation_output.totalDurationMean
    activitySequences[indexFilesTest[i]].totalDurationStandardDeviation = currentRunSimulation_output.totalDurationStDev
    activitySequences[indexFilesTest[i]].totalDurationMin = currentRunSimulation_output.totalDurationMin
    activitySequences[indexFilesTest[i]].totalDurationMax = currentRunSimulation_output.totalDurationMax
    activitySequences[indexFilesTest[i]].luckFactorMean = currentRunSimulation_output.luckFactorMean
    activitySequences[indexFilesTest[i]].trivialDecisionPercentageMean = currentRunSimulation_output.trivialDecisionPercentageMean


####  TEST NEURAL NETWORK MODEL ON TRAIN ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
for i in range(numberOfFilesTrain):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.indexActivitySequence = indexFilesTrain[i]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 0
    currentRunSimulation_input.policyType = "neuralNetworkModel"
    currentRunSimulation_input.decisionTool = neuralNetworkModel
    currentRunSimulation_input.activitySequences = activitySequences
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTrain[i]].totalDurationWithPolicy = currentRunSimulation_output.totalDurationMean

####  EVALUATION OF RESULTS OF TRAIN ACTIVITY SEQUENCES  ####
sumTotalDurationRandomTrain = 0
sumTotalDurationWithNeuralNetworkModelTrain = 0
for i in range(numberOfFilesTrain):
    sumTotalDurationRandomTrain += activitySequences[indexFilesTrain[i]].totalDurationMean
    sumTotalDurationWithNeuralNetworkModelTrain += activitySequences[indexFilesTrain[i]].totalDurationWithPolicy

####  TEST NEURAL NETWORK MODEL ON TEST ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
for i in range(numberOfFilesTest):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.indexActivitySequence = indexFilesTest[i]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 0
    currentRunSimulation_input.policyType = "neuralNetworkModel"
    currentRunSimulation_input.decisionTool = neuralNetworkModel
    currentRunSimulation_input.activitySequences = activitySequences
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTest[i]].totalDurationWithPolicy = currentRunSimulation_output.totalDurationMean

####  EVALUATION OF RESULTS OF TEST ACTIVITY SEQUENCES  ####
sumTotalDurationRandomTest = 0
sumTotalDurationWithNeuralNetworkModelTest = 0
for i in range(numberOfFilesTest):
    sumTotalDurationRandomTest += activitySequences[indexFilesTest[i]].totalDurationMean
    sumTotalDurationWithNeuralNetworkModelTest += activitySequences[indexFilesTest[i]].totalDurationWithPolicy

print("sumTotalDurationRandomTrain = " + str(sumTotalDurationRandomTrain))
print("sumTotalDurationWithNeuralNetworkModelTrain = " + str(sumTotalDurationWithNeuralNetworkModelTrain))
print("sumTotalDurationRandomTest = " + str(sumTotalDurationRandomTest))
print("sumTotalDurationWithNeuralNetworkModelTest = " + str(sumTotalDurationWithNeuralNetworkModelTest))

# compute computation time
t_end = time.time()
t_computation = t_end - t_start
print("t_computation = " + str(t_computation))


