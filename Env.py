import random
import statistics as st
from itertools import chain, combinations
import numpy as np


def powerset(listOfElements):
    s = list(listOfElements)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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
        self.activitySequence = []
        self.numberOfSimulationRuns = None
        self.timeDistribution = None
        self.purpose = None
        self.randomDecisionProbability = None
        self.policyType = None
        self.decisionTool = None
        self.numberOfResources = None
        self.numberOfActivitiesInStateVector = None
        self.stateVectorLength = None
        self.decisions_indexActivity = []
        self.rescaleFactorTime = None
        self.numberOfActivities = None

class runSimulation_output:
    def __init__(self):
        self.totalDurationMean = None
        self.totalDurationStDev = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.trivialDecisionPercentageMean = None
        self.stateActionPairsOfBestRun = []


def runSimulation(runSimulation_input):

    currentActivitySequence = runSimulation_input.activitySequence
    numberOfSimulationRuns = runSimulation_input.numberOfSimulationRuns
    timeDistribution = runSimulation_input.timeDistribution
    purpose = runSimulation_input.purpose
    randomDecisionProbability = runSimulation_input.randomDecisionProbability
    policyType = runSimulation_input.policyType
    decisionTool = runSimulation_input.decisionTool
    numberOfResources = runSimulation_input.numberOfResources
    numberOfActivitiesInStateVector = runSimulation_input.numberOfActivitiesInStateVector
    stateVectorLength = runSimulation_input.stateVectorLength
    decisions_indexActivity = runSimulation_input.decisions_indexActivity
    rescaleFactorTime = runSimulation_input.rescaleFactorTime
    numberOfActivities = runSimulation_input.numberOfActivities

    #print("start " + str(currentActivitySequence.fileName[:-4]))
    #print('------------------------------------------------------------------------------------------')

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
                    # verify that enough resources are available for every ready to start activity
                    enoughResourcesAreAvailable = True
                    for j in range(numberOfResources):
                        if currentActivity.requiredResources[j] > currentActivitySequence.availableResources[j]:
                            enoughResourcesAreAvailable = False
                            break
                    if enoughResourcesAreAvailable:
                        indexReadyToStartActivities.append(i)
            #print('indexReadyToStartActivities',indexReadyToStartActivities)

            # 1.2 check if the decision is trivial
            trivialDecision = True
            indexReadyToStartActivitiesInState = indexReadyToStartActivities[0:min(numberOfActivitiesInStateVector,len(indexReadyToStartActivities))]
            # compute powerset of decisions_indexActivity
            indexReadyToStartActivitiesPowerset = list(powerset(indexReadyToStartActivitiesInState))
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
                #print('not trivialDecision')

            numberOfDecisions += 1
            if trivialDecision:
                numberOfTrivialDecisions +=1
                #print('numberOfTrivialDecisions',numberOfTrivialDecisions)

            # 1.3 define activity conversion vector and resource conversion vector
            # initialise activityConversionVector and ResourceConversionVector
            activityConversionVector = [-1] * numberOfActivitiesInStateVector
            activityScores = []
            indexReadyToStartActivitiesInState = indexReadyToStartActivities[0:min(numberOfActivitiesInStateVector, len(indexReadyToStartActivities))]
            #print('indexReadyToStartActivitiesInState',indexReadyToStartActivitiesInState)

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
                        # print('resourceNeedForReadyToStartActivities',resourceNeedForReadyToStartActivities)

                # create resourceConversionVector
                indexResourcesGlobal = list(range(0,numberOfResources))
                indexResourcesGlobal_reordered = [x for _, x in sorted(zip(resourceNeedForReadyToStartActivities, indexResourcesGlobal), reverse=True)]
                resourceConversionVector = indexResourcesGlobal_reordered
                #print('resourceConversionVector',resourceConversionVector)

                # reorder activities depending on resource utilisation
                activityScores = [-1] * numberOfActivitiesInStateVector

                for i in range(len(indexReadyToStartActivitiesInState)):
                    for j in range(len(resourceConversionVector)):
                        resourceMultiplicator = 100 ** (numberOfResources-j-1)
                        #print('resourceMultiplicator',resourceMultiplicator)
                        resourceQuantity = currentActivitySequence.activities[indexReadyToStartActivitiesInState[i]].requiredResources[resourceConversionVector[j]]
                        activityScores[i] += 1 + resourceMultiplicator * resourceQuantity
                        #print('activityScores',activityScores)

                indexActivitiesGlobal = [-1] * numberOfActivitiesInStateVector
                indexActivitiesGlobal[0:len(indexReadyToStartActivitiesInState)] = indexReadyToStartActivitiesInState
                indexActivitiesGlobal_reordered = [x for _, x in sorted(zip(activityScores, indexActivitiesGlobal), reverse=True)]
                activityConversionVector = indexActivitiesGlobal_reordered
                #print('activityConversionVector',activityConversionVector)


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
            #print('randomDecisionAtThisStep',randomDecisionAtThisStep)


            if trivialDecision:    # if the decision is trivial, it does not matter how the priority values are assigned
                randomDecisionAtThisStep = True

            if randomDecisionAtThisStep:
                priorityValues = np.random.rand(numberOfActivitiesInStateVector)
                # print('randomDecisionAtThisStep')

            else:
                if policyType == "neuralNetworkModel":
                    currentState_readyToStartActivities = currentState_readyToStartActivities.reshape(-1, stateVectorLength)
                    #print('currentState_readyToStartActivities:',currentState_readyToStartActivities)
                    outputNeuralNetworkModel = decisionTool.predict(currentState_readyToStartActivities)
                    #print('outputNeuralNetworkModel:',outputNeuralNetworkModel)
                    priorityValues = np.zeros(numberOfActivitiesInStateVector)
                    for i in range(len(outputNeuralNetworkModel)):
                        # priorityValues[i] = outputNeuralNetworkModel[0,i]
                        priorityValues = outputNeuralNetworkModel[0]
                        #print('priorityValues:',priorityValues)
                elif policyType == "otherPolicy1":
                    print("generate priority values with other policy 1")
                elif policyType == "otherPolicy2":
                    print("generate priority values with other policy 2")
                else:
                    print("policy name not existing")



            # reorder list according to priority
            decisions_indexActivity_reordered = [x for _, x in sorted(zip(priorityValues,decisions_indexActivity), reverse=True)]
            #print('decisions_indexActivity_reordered)',decisions_indexActivity_reordered)

            # if not randomDecisionAtThisStep:
            #     print('not randomdecision##############################################################################################')
            #     print('decisions_indexActivity_reordered',decisions_indexActivity_reordered)

            # use the priority values to start new activities
            currentAction = np.zeros([numberOfActivitiesInStateVector])
            indexStartedActivities = []
            # consider the decision one by one in reordered list
            for indexActivityToStartLocal in decisions_indexActivity_reordered:
                indexActivityToStartGlobal = activityConversionVector[indexActivityToStartLocal]
                #print('indexActivityToStartGlobal',indexActivityToStartGlobal)

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

            #print('currentaction',currentAction)


            # 1.8 if the purpose is to generate training data, save the current state action pair
            if purpose == "generateData" and trivialDecision == False:
                currentStateActionPair = stateActionPair()
                currentStateActionPair.state = currentState_readyToStartActivities
                #print('currentState_readyToStartActivities',currentState_readyToStartActivities)
                currentStateActionPair.action = currentAction
                #print('currentAction',currentAction)
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
        #print('trivialDecisionPercentages',trivialDecisionPercentages)

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
            indexBestRun = totalDurations.index(totalDurationMin)
            currentRunSimulation_output.stateActionPairsOfBestRun = stateActionPairsOfRuns[indexBestRun]


    #print("end " + str(currentActivitySequence.fileName[:-4]))
    #print('-------------------------------------------------------------')


    return currentRunSimulation_output