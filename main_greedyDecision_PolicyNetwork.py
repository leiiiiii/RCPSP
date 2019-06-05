#!/usr/bin/python3
# -*- coding: utf-8 -*-
import openpyxl
# import xlsxwriter

import os    #to get of directory paths
import glob    #to read all files in directory
import re    #to decompose strings
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression
from NN_Model import createNeuralNetworkModel
from collections import Counter
import numpy as np
import random
import statistics as st    #statistical functions
import multiprocessing
from itertools import chain, combinations    #generation of powersets
import time

#DEBUGGING
#...

#TODO
#test CPU parallelisation -> it is working but it stops sometimes when using pool.map
#test GPU parallelization -> probably not possible because the GPU do not accept parallel tasks
#tensorflow GPU with float 16bit
#run generate data and train in parallel avoiding that the buffer becomes too big
#generate extra activity sequences
#assign priority values also to ready to start activities outside the state vector
#introduce check for existing neural network in path
#test different learning parameters
#test with stochastic times and luck factors
#global (SL) + local (RL)
#separate codes in different scripts if the code becomes too big (main, runSimulation)
#try discount factor for resource utilization in the future
#also test value network
#test different learning libraries (tf, tflearn, keras, ...)

#MAYBE TO BE MODIFIED
#take same number of random files from each class
#tools for better visualization (GUI simulation)
#decision policy based on "prefer activities with a lot of resource of a certain type required"


##################################################    FUNCTIONS    ##################################################
# run simulation
def runSimulation(runSimulation_input):

    currentIndexActivitySequence = runSimulation_input.indexActivitySequence
    numberOfSimulationRuns = runSimulation_input.numberOfSimulationRuns
    timeDistribution = runSimulation_input.timeDistribution
    purpose = runSimulation_input.purpose
    randomDecisionProbability = runSimulation_input.randomDecisionProbability
    policyType = runSimulation_input.policyType
    decisionTool = runSimulation_input.decisionTool

    currentActivitySequence = activitySequences[currentIndexActivitySequence]

    print("start " + str(currentActivitySequence.fileName[:-4]))

    # reset variables for the series of runs
    indexSimulationRun = 0
    # reset lists for the series of runs
    totalDurations = []
    luckFactors = []
    trivialDecisionPercentages = []
    stateActionPairsOfRuns = []
    while indexSimulationRun < numberOfSimulationRuns:
        # reset variables for run
        sumActivityDuration = 0
        step = 0
        numberOfDecisions = 0
        numberOfTrivialDecisions = 0
        # reset lists for run
        if purpose == "generateData":
            currentStateActionPairsOfRun = []
        # reset dynamic variables of classes for run
        currentActivitySequence.availableResources = currentActivitySequence.totalResources[:]
        currentActivitySequence.virtualTime = 0
        for i in range(len(currentActivitySequence.activities)):
            currentActivitySequence.activities[i].withToken = False
            currentActivitySequence.activities[i].idleToken = False
            currentActivitySequence.activities[i].numberOfCompletedPreviousActivities = 0
            currentActivitySequence.activities[i].remainingTime = 0
            currentActivitySequence.activities[i].processedTime = 0
            currentActivitySequence.activities[i].seizedResources = [0] * numberOfResources

        # set startActivities ready to start
        for indexStartActivity in currentActivitySequence.indexStartActivities:
            currentActivitySequence.activities[indexStartActivity].withToken = True
            currentActivitySequence.activities[indexStartActivity].idleToken = True

        # start simulation
        simulationRunFinished = False
        while simulationRunFinished == False:    # if there are some token left in some activities
            step += 1

            ## STEP 1 ##
            # 1.1 find activities ready to start
            indexReadyToStartActivities = []
            for i, currentActivity in enumerate(currentActivitySequence.activities):
                if (currentActivity.withToken and currentActivity.idleToken and currentActivity.numberOfCompletedPreviousActivities == currentActivity.numberOfPreviousActivities):
                    # verify that enough resources are available to start
                    enoughResourcesAreAvailable = True
                    for j in range(numberOfResources):
                        if currentActivity.requiredResources[j] > currentActivitySequence.availableResources[j]:
                            enoughResourcesAreAvailable = False
                            break
                    if enoughResourcesAreAvailable:
                        indexReadyToStartActivities.append(i)

            # 1.2 check if the decision is trivial
            trivialDecision = True
            # compute powerset of decisions_indexActivity
            indexReadyToStartActivitiesPowerset = list(powerset(indexReadyToStartActivities))
            # find feasible combined decisions_indexActivity (only resource check needed)
            feasibleCombinedDecisions_indexActivity = []
            for i in reversed(range(len(indexReadyToStartActivitiesPowerset))):
                currentDecision = list(indexReadyToStartActivitiesPowerset[i])
                decisionIsASubsetOfFeasibleDecision = False
                for j,feasibleDecisionAlreadyInList in enumerate(feasibleCombinedDecisions_indexActivity):
                    if len(set(currentDecision) - set(feasibleDecisionAlreadyInList)) == 0:
                        decisionIsASubsetOfFeasibleDecision = True
                        break
                if decisionIsASubsetOfFeasibleDecision == False:
                    # verify that enough resources are available to start all the activities
                    totalRequiredResources = [0] * numberOfResources
                    for indexCurrentActivity in currentDecision:
                        for j in range(numberOfResources):
                            totalRequiredResources[j] += currentActivitySequence.activities[indexCurrentActivity].requiredResources[j]
                    enoughResourcesAreAvailable = True
                    for j in range(numberOfResources):
                        if totalRequiredResources[j] > currentActivitySequence.availableResources[j]:
                            enoughResourcesAreAvailable = False
                            break
                    if enoughResourcesAreAvailable:
                        feasibleCombinedDecisions_indexActivity.append(currentDecision)
            if len(feasibleCombinedDecisions_indexActivity) > 1:
                trivialDecision = False

            numberOfDecisions += 1
            if trivialDecision:
                numberOfTrivialDecisions +=1

            # 1.3 define activity conversion vector and resource conversion vector
            # initialise activityConversionVector and ResourceConversionVector
            activityConversionVector = [-1] * numberOfActivitiesInStateVector
            activityScores = []
            indexReadyToStartActivitiesInState = indexReadyToStartActivities[0:min(numberOfActivitiesInStateVector, len(indexReadyToStartActivities))]
            if trivialDecision:
                # no conversion needed
                resourceConversionVector = list(range(0,numberOfResources))
                for i in range(len(indexReadyToStartActivitiesInState)):
                    activityConversionVector[i] = indexReadyToStartActivitiesInState[i]
            else:
                # conversion is required
                # find most critical resources (i.e. required resources to start the ready to start activities normalized by the total resources)
                resourceNeedForReadyToStartActivities = [0] * numberOfResources
                for i in indexReadyToStartActivities:
                    for j in range(numberOfResources):
                        resourceNeedForReadyToStartActivities[j] += currentActivitySequence.activities[i].requiredResources[j] / currentActivitySequence.totalResources[j]
                # create resourceConversionVector
                indexResourcesGlobal = list(range(0,numberOfResources))
                indexResourcesGlobal_reordered = [x for _, x in sorted(zip(resourceNeedForReadyToStartActivities, indexResourcesGlobal), reverse=True)]
                resourceConversionVector = indexResourcesGlobal_reordered
                # reorder activities depending on resource utilisation
                activityScores = [-1] * numberOfActivitiesInStateVector
                for i in range(len(indexReadyToStartActivitiesInState)):
                    for j in range(len(resourceConversionVector)):
                        resourceMultiplicator = 100 ** (numberOfResources-j-1)
                        resourceQuantity = currentActivitySequence.activities[indexReadyToStartActivitiesInState[i]].requiredResources[resourceConversionVector[j]]
                        activityScores[i] += 1 + resourceMultiplicator * resourceQuantity

                indexActivitiesGlobal = [-1] * numberOfActivitiesInStateVector
                indexActivitiesGlobal[0:len(indexReadyToStartActivitiesInState)] = indexReadyToStartActivitiesInState
                indexActivitiesGlobal_reordered = [x for _, x in sorted(zip(activityScores, indexActivitiesGlobal), reverse=True)]
                activityConversionVector = indexActivitiesGlobal_reordered

            # 1.4 normalized state vector and matrix are created
            currentState_readyToStartActivities = []
            if trivialDecision == False:
                currentState_readyToStartActivities = np.zeros([stateVectorLength])
                for i, indexActivity in enumerate(activityConversionVector):
                    if indexActivity != -1:
                        currentState_readyToStartActivities[0+i*(1+numberOfResources)] = currentActivitySequence.activities[indexActivity].time * rescaleFactorTime
                        for j in range(numberOfResources):
                            currentState_readyToStartActivities[1 + j + i * (1 + numberOfResources)] = currentActivitySequence.activities[indexActivity].requiredResources[resourceConversionVector[j]] / currentActivitySequence.totalResources[resourceConversionVector[j]]
                for j in range(numberOfResources):
                    currentState_readyToStartActivities[numberOfActivitiesInStateVector + numberOfActivitiesInStateVector * numberOfResources + j] = currentActivitySequence.availableResources[resourceConversionVector[j]] / currentActivitySequence.totalResources[resourceConversionVector[j]]
            # (optional: add information about the future resource utilisation)
            # determine the earliest starting point of each activity considering the problem without resource constraints and deterministic
            # currentState_futureResourceUtilisation = np.zeros([numberOfResources, timeHorizon])

            # 1.5 Use the policy and the decision tool to define which tokens can begin the correspondent activity or remain idle
            randomDecisionAtThisStep = (random.random() < randomDecisionProbability)
            if trivialDecision:    # if the decision is trivial, it does not matter how the priority values are assigned
                randomDecisionAtThisStep = True
            if randomDecisionAtThisStep:
                priorityValues = np.random.rand(numberOfActivitiesInStateVector)
            else:
                if policyType == "neuralNetworkModel":
                    currentState_readyToStartActivities = currentState_readyToStartActivities.reshape(-1, stateVectorLength)
                    outputNeuralNetworkModel = decisionTool.predict(currentState_readyToStartActivities)
                    priorityValues = np.zeros(numberOfActivitiesInStateVector)
                    for i in range(len(outputNeuralNetworkModel)):
                        priorityValues[i] = outputNeuralNetworkModel[0,i]
                elif policyType == "otherPolicy1":
                    print("generate priority values with other policy 1")
                elif policyType == "otherPolicy2":
                    print("generate priority values with other policy 2")
                else:
                    print("policy name not existing")

            # reorder list according to priority
            decisions_indexActivity_reordered = [x for _, x in sorted(zip(priorityValues,decisions_indexActivity), reverse=True)]

            # use the priority values to start new activities
            currentAction = np.zeros([numberOfActivitiesInStateVector])
            indexStartedActivities = []
            # consider the decision one by one in reordered list
            for indexActivityToStartLocal in decisions_indexActivity_reordered:
                indexActivityToStartGlobal = activityConversionVector[indexActivityToStartLocal]
                if indexActivityToStartGlobal != -1:
                    currentActivity = currentActivitySequence.activities[indexActivityToStartGlobal]
                    if currentActivity.withToken and currentActivity.idleToken and currentActivity.numberOfCompletedPreviousActivities == currentActivity.numberOfPreviousActivities:
                        # verify that enough resources are available to start
                        enoughResourcesAreAvailable = True
                        for i in range(numberOfResources):
                            if currentActivity.requiredResources[i] > currentActivitySequence.availableResources[i]:
                                enoughResourcesAreAvailable = False
                                break
                        if enoughResourcesAreAvailable:
                            currentActivitySequence.activities[indexActivityToStartGlobal].idleToken = False

                            # 1.6 Set remaining time for the starting activity
                            if timeDistribution == "deterministic":
                                currentActivitySequence.activities[indexActivityToStartGlobal].remainingTime = currentActivitySequence.activities[indexActivityToStartGlobal].time
                                sumActivityDuration += currentActivitySequence.activities[indexActivityToStartGlobal].remainingTime

                            # 1.7 seize resources
                            for i in range(numberOfResources):
                                currentActivitySequence.activities[indexActivityToStartGlobal].seizedResources[i] = currentActivitySequence.activities[indexActivityToStartGlobal].requiredResources[i]
                                currentActivitySequence.availableResources[i] -= currentActivitySequence.activities[indexActivityToStartGlobal].requiredResources[i]

                            # update the action vector with the activity that has been just started
                            currentAction[indexActivityToStartLocal] = 1
                            indexStartedActivities.append(indexActivityToStartGlobal)

            # 1.8 if the purpose is to generate training data, save the current state action pair
            if purpose == "generateData" and trivialDecision == False:
                currentStateActionPair = stateActionPair()
                currentStateActionPair.state = currentState_readyToStartActivities
                currentStateActionPair.action = currentAction
                currentStateActionPairsOfRun.append(currentStateActionPair)

            ## STEP 2 ##
            # 2.1 find out when the next event (activity end) occurs
            smallestRemainingTime = 1e300
            indexActiveActivities = []
            for i in range(numberOfActivities):
                if currentActivitySequence.activities[i].withToken and currentActivitySequence.activities[i].idleToken == False:
                    indexActiveActivities.append(i)
                    if currentActivitySequence.activities[i].remainingTime < smallestRemainingTime:
                        smallestRemainingTime = currentActivitySequence.activities[i].remainingTime
            # 2.2 find next finishing activities
            indexNextFinishingActivities = []
            for i in indexActiveActivities:
                if currentActivitySequence.activities[i].remainingTime == smallestRemainingTime:
                    indexNextFinishingActivities.append(i)

            # 2.3 jump forward to activity end
            currentActivitySequence.virtualTime += smallestRemainingTime
            for i in indexActiveActivities:
                currentActivitySequence.activities[i].remainingTime -= smallestRemainingTime
                currentActivitySequence.activities[i].processedTime += smallestRemainingTime

            ## STEP 3 ##
            # for each just finished activity:
            for i in indexNextFinishingActivities:
                # 3.1 find following activities
                indexFollowingActivities = currentActivitySequence.activities[i].indexFollowingActivities
                # 3.2 for each following activity, increment the numberOfCompletedPreviousActivities and, if there is no token already in the following activity, add an idle token.
                for j in indexFollowingActivities:
                    currentActivitySequence.activities[j].numberOfCompletedPreviousActivities += 1
                    if currentActivitySequence.activities[j].withToken == False:
                        currentActivitySequence.activities[j].withToken = True
                        currentActivitySequence.activities[j].idleToken = True
                # 3.3 delete token from just finished activity
                currentActivitySequence.activities[i].withToken = False
                currentActivitySequence.activities[i].idleToken = False
                # 3.4 release resources back to the resource pool
                currentActivitySequence.activities[i].seizedResources = [0] * numberOfResources
                for j in range(numberOfResources):
                    currentActivitySequence.availableResources[j] += currentActivitySequence.activities[i].requiredResources[j]

            ## STEP 4 ##
            # check if all activities are completed (i.e. no more token)
            simulationRunFinished = True
            for i in range(numberOfActivities):
                if currentActivitySequence.activities[i].withToken:
                    simulationRunFinished = False
                    break

        totalDuration = currentActivitySequence.virtualTime
        luckFactor = sumActivityDuration / sum(a.time for a in currentActivitySequence.activities)
        trivialDecisionPercentage = numberOfTrivialDecisions / numberOfDecisions

        totalDurations.append(totalDuration)
        luckFactors.append(luckFactor)
        trivialDecisionPercentages.append(trivialDecisionPercentage)

        if purpose == "generateData":
            stateActionPairsOfRuns.append(currentStateActionPairsOfRun)

        # increment the index for the simulation run at the end of the loop
        indexSimulationRun += 1

    totalDurationMean = st.mean(totalDurations)
    totalDurationStDev = None
    if numberOfSimulationRuns != 1:
        totalDurationStDev = st.stdev(totalDurations)
    totalDurationMin = min(totalDurations)
    totalDurationMax = max(totalDurations)
    luckFactorMean = st.mean(luckFactors)
    trivialDecisionPercentageMean = st.mean(trivialDecisionPercentages)

    currentRunSimulation_output = runSimulation_output()
    currentRunSimulation_output.totalDurationMean = totalDurationMean
    currentRunSimulation_output.totalDurationStDev = totalDurationStDev
    currentRunSimulation_output.totalDurationMin = totalDurationMin
    currentRunSimulation_output.totalDurationMax = totalDurationMax
    currentRunSimulation_output.luckFactorMean = luckFactorMean
    currentRunSimulation_output.trivialDecisionPercentageMean = trivialDecisionPercentageMean

    # submit the stateActionPairs of the best run, if the standard deviation of the duration is not zero
    if purpose == "generateData":
        if totalDurationStDev != 0:
            indexBestRun = totalDurations.index(totalDurationMax)
            currentRunSimulation_output.stateActionPairsOfBestRun = stateActionPairsOfRuns[indexBestRun]

    print("end " + str(currentActivitySequence.fileName[:-4]))

    return currentRunSimulation_output


# return powerset (Potenzmenge) of a set "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
def powerset(listOfElements):
    s = list(listOfElements)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


##################################################    CLASSES    ##################################################
class activitySequence:
    def __init__(self):
        # static (do not change during simulation)
        self.index = None
        self.fileName = None
        self.numberOfResources = None
        self.totalResources = []
        self.numberOfActivities = None
        self.activities = []
        self.indexStartActivities = []
        # dynamic (change during simulation)
        self.availableResources = []
        self.totalDurationMean = None
        self.totalDurationStandardDeviation = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.totalDurationWithPolicy = None
        self.trivialDecisionPercentageMean = None

class activity:
    def __init__(self):
        # static (do not change during simulation)
        self.index = None
        self.time = None    #expected value. Only deterministic component. The distribution is given as an argument in the function run simulation.
        self.requiredResources = []
        self.numberOfPreviousActivities = 0
        self.indexFollowingActivities = []
        # dynamic (change during simulation)
        self.withToken = None
        self.idleToken = None
        self.numberOfCompletedPreviousActivities = None
        self.remainingTime = None    #time to activity end
        self.processedTime = None    #time elapsed from the beginning of the activity
        self.seizedResources = []

class stateActionPair:
    def __init__(self):
        self.state = None
        self.action = None

class runSimulation_input:
    def __init__(self):
        self.indexActivitySequence = None
        self.numberOfSimulationRuns = None
        self.timeDistribution = None
        self.purpose = None
        self.randomDecisionProbability = None
        self.policyType = None
        self.decisionTool = None

class runSimulation_output:
    def __init__(self):
        self.totalDurationMean = None
        self.totalDurationStDev = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.trivialDecisionPercentageMean = None
        self.stateActionPairsOfBestRun = []


##################################################    MAIN    ##################################################
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
numberOfSimulationRunsToGenerateData = 10
numberOfSimulationRunsToTestPolicy = 1
# train parameters
percentageOfFilesTest = 0.1
importExistingNeuralNetworkModel = False
numberOfEpochs = 3
learningRate = 0.001
# paths
relativePath = os.path.dirname(__file__)
absolutePathProjects = relativePath + "/RG30_merged/"
absolutePathExcelOutput = relativePath + "/Benchmark.xlsx"
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
        next(f)
        firstLine = f.readline()    # information about numberOfActivities and numberOfResourceTypes
        firstLineDecomposed = re.split(" +", firstLine)
        numberOfActivities = (int(firstLineDecomposed[1])-2)    # the first and last dummy activity do not count
        currentActivitySequence.numberOfActivities = numberOfActivities
        # print("numberOfActivities = " + str(currentActivitySequence.numberOfActivities))
        secondLine = f.readline()   # information about total available resources
        secondLineDecomposed = re.split(" +", secondLine)
        secondLineDecomposed=''.join(secondLineDecomposed).strip('\n')#only string have attributes of strip and delete '\n' at the end
        numberOfResources = 0
        # print(len(secondLineDecomposed))
        secondLineDecomposed=[int(secondLineDecomposed)]
        # print(secondLineDecomposed)
        for totalResources in secondLineDecomposed:
            numberOfResources += 1
            currentActivitySequence.totalResources.append(int(totalResources))
            # print(currentActivitySequence.totalResources)
        currentActivitySequence.numberOfResources = numberOfResources
        next(f)
        thirdLine = f.readline()   # information about starting activities
        thirdLineDecomposed = re.split(" +", thirdLine)
        for IdActivity in thirdLineDecomposed[4:-1]:
            currentActivitySequence.indexStartActivities.append(int(IdActivity)-2)
        #print("indexStartingActivities = " + str(currentActivitySequence.indexStartActivities))
        line = f.readline()
        while line:
            #print(line, end="")
            lineDecomposed = re.split(" +", line)
            if int(lineDecomposed[1]) == 0:
                break
            else:
                currentActivity = activity()
                currentActivity.time = int(lineDecomposed[1])
                currentActivity.requiredResources = [ int(lineDecomposed[2])]
                for IdFollowingActivity in lineDecomposed[4:-1]:
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

# ####  GENERATE TRAINING DATA USING RANDOM DECISIONS  ####
# # run simulations on train activity sequences with random policy and multiprocessing
# runSimulation_inputs = []
# for i in range(numberOfFilesTrain):
#     currentRunSimulation_input = runSimulation_input()
#     currentRunSimulation_input.indexActivitySequence = indexFilesTrain[i]
#     currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
#     currentRunSimulation_input.timeDistribution = timeDistribution
#     currentRunSimulation_input.purpose = "generateData"
#     currentRunSimulation_input.randomDecisionProbability = 1
#     currentRunSimulation_input.policyType = None
#     currentRunSimulation_input.decisionTool = None
#     runSimulation_inputs.append(currentRunSimulation_input)
# pool = multiprocessing.Pool(processes=numberOfCpuProcessesToGenerateData, maxtasksperchild=maxTasksPerChildToGenerateData)
# runSimulation_outputs = pool.map(runSimulation, runSimulation_inputs)
# # assign simulation results to activity sequences and append training data
# for i in range(numberOfFilesTrain):
#     activitySequences[indexFilesTrain[i]].totalDurationMean = runSimulation_outputs[i].totalDurationMean
#     activitySequences[indexFilesTrain[i]].totalDurationStandardDeviation = runSimulation_outputs[i].totalDurationStDev
#     activitySequences[indexFilesTrain[i]].totalDurationMin = runSimulation_outputs[i].totalDurationMin
#     activitySequences[indexFilesTrain[i]].totalDurationMax = runSimulation_outputs[i].totalDurationMax
#     activitySequences[indexFilesTrain[i]].luckFactorMean = runSimulation_outputs[i].luckFactorMean
#     activitySequences[indexFilesTrain[i]].trivialDecisionPercentageMean = runSimulation_outputs[i].trivialDecisionPercentageMean
#     for currentStateActionPair in runSimulation_outputs[i].stateActionPairsOfBestRun:
#         states.append(currentStateActionPair.state)
#         actions.append(currentStateActionPair.action)



####  TRAIN MODEL USING TRAINING DATA  ####
# look for existing model
if importExistingNeuralNetworkModel:
    neuralNetworkModelAlreadyExists = False
    print("check if a neural network model exists")
    if neuralNetworkModelAlreadyExists:
        print("import neural network model exists")
    else:
        neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
else:
    neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)

neuralNetworkModel.fit({"input": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_step=500, show_metric=True, run_id="trainNeuralNetworkModel")

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
    currentRunSimulation_output = runSimulation(currentRunSimulation_input)
    activitySequences[indexFilesTest[i]].totalDurationMean = currentRunSimulation_output.totalDurationMean
    activitySequences[indexFilesTest[i]].totalDurationStandardDeviation = currentRunSimulation_output.totalDurationStDev
    activitySequences[indexFilesTest[i]].totalDurationMin = currentRunSimulation_output.totalDurationMin
    activitySequences[indexFilesTest[i]].totalDurationMax = currentRunSimulation_output.totalDurationMax
    activitySequences[indexFilesTest[i]].luckFactorMean = currentRunSimulation_output.luckFactorMean
    activitySequences[indexFilesTest[i]].trivialDecisionPercentageMean = currentRunSimulation_output.trivialDecisionPercentageMean


# ####  CREATE BENCHMARK WITH RANDOM DECISIONS ALSO WITH TEST ACTIVITY SEQUENCES  ####
# # run simulations on test activity sequences with random policy and multiprocessing
# runSimulation_inputs = []
# for i in range(numberOfFilesTest):
#     currentRunSimulation_input = runSimulation_input()
#     currentRunSimulation_input.indexActivitySequence = indexFilesTest[i]
#     currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
#     currentRunSimulation_input.timeDistribution = timeDistribution
#     currentRunSimulation_input.purpose = "testPolicy"
#     currentRunSimulation_input.randomDecisionProbability = 1
#     currentRunSimulation_input.policyType = None
#     currentRunSimulation_input.decisionTool = None
#     runSimulation_inputs.append(currentRunSimulation_input)
# pool = multiprocessing.Pool(processes=numberOfCpuProcessesToGenerateData, maxtasksperchild=maxTasksPerChildToGenerateData)
# runSimulation_outputs = pool.map(runSimulation, runSimulation_inputs)
# # assign simulation results to activity sequences
# for i in range(numberOfFilesTest):
#     activitySequences[indexFilesTest[i]].totalDurationMean = runSimulation_outputs[i].totalDurationMean
#     activitySequences[indexFilesTest[i]].totalDurationStandardDeviation = runSimulation_outputs[i].totalDurationStDev
#     activitySequences[indexFilesTest[i]].totalDurationMin = runSimulation_outputs[i].totalDurationMin
#     activitySequences[indexFilesTest[i]].totalDurationMax = runSimulation_outputs[i].totalDurationMax
#     activitySequences[indexFilesTest[i]].luckFactorMean = runSimulation_outputs[i].luckFactorMean
#     activitySequences[indexFilesTest[i]].trivialDecisionPercentageMean = runSimulation_outputs[i].trivialDecisionPercentageMean

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




# # write output to excel with xlsxwriter
# wb = xlsxwriter.Workbook(absolutePathExcelOutput)
# ws = wb.add_worksheet("J30_totalDurations")
# ws.set_column("A:A", 22)
# ws.write(0, 0, "number of simulation runs")
# ws.write(0, 1, numberOfSimulationRuns)
# ws.write(1, 0, "computation time")
# ws.write(1, 1, t_computation)
# ws.write(2, 1, "solution random")
# ws.write(3, 0, "activity sequence name")
# ws.write(3, 1, "E[T]")
# ws.write(3, 2, "StDev[T]")
# ws.write(3, 3, "min[T]")
# ws.write(3, 4, "max[T]")
# ws.write(3, 5, "P[trivial decision]")
# ws.write(3, 6, "min=max")
# for i in indexFilesTrain:
#     ws.write(i+4, 0, activitySequences[i].fileName[:-4])
#     ws.write(i+4, 1, activitySequences[i].totalDurationMean)
#     ws.write(i+4, 2, activitySequences[i].totalDurationStandardDeviation)
#     ws.write(i+4, 3, activitySequences[i].totalDurationMin)
#     ws.write(i+4, 4, activitySequences[i].totalDurationMax)
#     ws.write(i+4, 5, activitySequences[i].trivialDecisionPercentageMean)
#     if activitySequences[i].totalDurationMin == activitySequences[i].totalDurationMax:
#         ws.write(i+4, 6, "1")
#     else:
#         ws.write(i+4, 6, "")
#
# wb.close()

