import random
import statistics as st
from itertools import chain, combinations
import numpy as np
import itertools


#enumerate all possible action
def possibleAction(numberOfAction):
    Action = np.zeros((numberOfAction, numberOfAction), int)
    for i in range(numberOfAction):
        for number in range(i + 1):
            Action[i][number] = 1
    A_output = np.zeros((1,numberOfAction))
    for a in Action:
        list2=list(itertools.permutations(a))
        list2 = np.asarray(list2)
        list2 = np.unique(list2, axis=0)
        A_output = np.concatenate((A_output, list2), axis=0)
    return A_output

def powerset(listOfElements):
    s = list(listOfElements)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def actionsequence(Elements):
    a = combinations(range(1),2)


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
        self.totalDurationWithHeuristic = None
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

class stateActionPossibilityPair:
    def __init__(self):
        self.state = None
        self.actionPossibility = None


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
        self.timeHorizon = None

class runSimulation_output:
    def __init__(self):
        self.totalDurationMean = None
        self.totalDurationStDev = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.trivialDecisionPercentageMean = None
        self.stateActionPairsOfBestRun = []
        self.stateActionPossibilityPairsOfBestRun = []


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
    timeHorizon = runSimulation_input.timeHorizon

    #print("start " + str(currentActivitySequence.fileName[:-4]))
    #print('------------------------------------------------------------------------------------------')

    # reset variables for the series of runs
    indexSimulationRun = 0
    # reset lists for the series of runs
    totalDurations = []
    luckFactors = []
    trivialDecisionPercentages = []
    stateActionPairsOfRuns = []
    stateActionPossibilityPairsOfRun = []
    possibleactions = possibleAction(numberOfActivitiesInStateVector)


    while indexSimulationRun < numberOfSimulationRuns:
        # reset variables for run
        sumActivityDuration = 0
        step = 0
        numberOfDecisions = 0
        numberOfTrivialDecisions = 0

        # reset lists for run
        if purpose == "generateData":
            currentStateActionPairsOfRun = []
            currentStateActionPossibilityPairsOfRun = []

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
                #print('resourceConversionvector',resourceConversionVector)


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
            #print(activityConversionVector)


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

                currentState_readyToStartActivities.append(currentStateFuturnResourceUtilisation)


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
                    #print('outputNeuralNetworkModel',outputNeuralNetworkModel)
                    a = list(outputNeuralNetworkModel[0])
                    actionsindex = a.index(max(a))
                    priorityValues = possibleactions[actionsindex]

                    #priorityValues = outputNeuralNetworkModel[0]

                    print('priorityValues:',priorityValues)

                elif policyType == "heuristic":
                    #print("generate priority values with most critical resource")
                    priorityValues = [1, 0.8, 0.6, 0.4, 0.2, 0]

                elif policyType == "otherPolicy2":
                    print("generate priority values with other policy 2")
                else:
                    print("policy name not existing")


            # reorder list according to priority
            decisions_indexActivity_reordered = [x for _, x in sorted(zip(priorityValues,decisions_indexActivity), reverse=True)]

            # use the priority values to start new activities
            currentAction = np.zeros(numberOfActivitiesInStateVector)
            currentActionPossibility = np.zeros(64)
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
            #print('currentAction',currentAction)

            index = list(np.where(possibleactions==currentAction)[0])
            for value in index:
                if index.count(value)==numberOfActivitiesInStateVector:
                    indexlocation = value

            currentActionPossibility[indexlocation] = 1


            # 1.8 if the purpose is to generate training data, save the current state action pair
            if purpose == "generateData" and trivialDecision == False:
                currentStateActionPair = stateActionPair()
                currentStateActionPossibilityPair = stateActionPossibilityPair()

                currentStateActionPossibilityPair.state = currentState_readyToStartActivities
                currentStateActionPossibilityPair.actionPossibility = currentActionPossibility

                currentStateActionPair.state = currentState_readyToStartActivities
                currentStateActionPair.action = currentAction

                currentStateActionPairsOfRun.append(currentStateActionPair)
                currentStateActionPossibilityPairsOfRun.append(currentStateActionPossibilityPair)


            ## STEP 2 ##
            # 2.1 find out when the next event (activity end) occurs
            smallestRemainingTime = 1e300
            indexActiveActivities = []
            for i in range(numberOfActivities):
                if currentActivitySequence.activities[i].withToken and currentActivitySequence.activities[i].idleToken == False:
                    indexActiveActivities.append(i)

                    if currentActivitySequence.activities[i].remainingTime < smallestRemainingTime:
                        smallestRemainingTime = currentActivitySequence.activities[i].remainingTime

            #print('indexActiveActivities', indexActiveActivities)

#-----------------------------------------------------------------------------------------------------------------------------------------
            #generate timeHorizonMatrix for active activities
            timeHorizonMatrix = np.zeros((len(indexActiveActivities), timeHorizon))
            timeUnitmatrix=[x for x in range(len(indexActiveActivities))]
            remainingtimeList = []
            for i in indexActiveActivities:
                remainingtimeList.append(currentActivitySequence.activities[i].remainingTime)
            for value in remainingtimeList:
                if value > timeHorizon:
                    value = timeHorizon
            maximaltimeHorizon=max(remainingtimeList)
            for (i,j) in zip(timeUnitmatrix,remainingtimeList):
                timeHorizonMatrix[i][0:j]=1


            #generate resourceUtilizationMatrix for active activities
            resourcematrix = np.zeros((1, numberOfResources))
            for i in indexActiveActivities:
                a=[currentActivitySequence.activities[i].requiredResources]
                resourcematrix=np.concatenate((resourcematrix,a),axis=0)
            resourcematrix=resourcematrix[1:]
            resourceUtilizationMatrix = resourcematrix.T


            #currentState_futureResourceUtilisation for active activity generated
            currentState_futureResourceUtilisation = np.dot(resourceUtilizationMatrix,timeHorizonMatrix)

#-----------------------------------------------------------------------------------------------------------------------------------------

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

            #print('indexFollowingActivities',indexFollowingActivities)


            #-------------------------------------------------------------------------------------------------------------------------------
            # generate timeHorizonMatrix for following activities (timeHorizon starts maximaltimeHOrizon)
            timeHorizonMatrixforFollowing = np.zeros((len(indexFollowingActivities), timeHorizon))
            if len(indexFollowingActivities) > 1:
                timeUnitmatrixforFollowing = [x for x in range(len(indexFollowingActivities))]
                timeListforFollowing = []
                for i in indexFollowingActivities:
                    timeListforFollowing.append(currentActivitySequence.activities[i].time)

                for value in timeListforFollowing:
                    if value+maximaltimeHorizon>timeHorizon:
                        value = timeHorizon-maximaltimeHorizon

                for (i, j) in zip(timeUnitmatrixforFollowing, timeListforFollowing):
                    timeHorizonMatrixforFollowing[i][maximaltimeHorizon:j] = 1

            elif len(indexFollowingActivities)==1:
                for i in indexFollowingActivities:
                    index=currentActivitySequence.activities[i].time
                if index+maximaltimeHorizon>10:
                    index = timeHorizon-maximaltimeHorizon
                    timeHorizonMatrixforFollowing[0][maximaltimeHorizon:index] = 1

            else:
                break


            # generate resourceUtilizationMatrix for following activities
            resourcematrixforFollowing = np.zeros((1, numberOfResources))
            if len(indexFollowingActivities) > 1:
                for i in indexFollowingActivities:
                    a = [currentActivitySequence.activities[i].requiredResources]
                    resourcematrixforFollowing = np.concatenate((resourcematrixforFollowing, a), axis=0)
                resourcematrixforFollowing = resourcematrixforFollowing[1:]
                resourceUtilizationMatrixforFollowing = resourcematrixforFollowing.T

            elif len(indexFollowingActivities)==1:
                for i in indexFollowingActivities:
                    currentActivitySequence.activities[i].requiredResources=np.array(currentActivitySequence.activities[i].requiredResources)
                    resourceUtilizationMatrixforFollowing=currentActivitySequence.activities[i].requiredResources.reshape((numberOfResources,1))


            # currentState_futureResourceUtilisation for following activities generated
            currentState_futureResourceUtilisation_forFollowing = np.dot(resourceUtilizationMatrixforFollowing, timeHorizonMatrixforFollowing)

            currentState_futureResourceUtilisation = np.add(currentState_futureResourceUtilisation,currentState_futureResourceUtilisation_forFollowing)

            resourceConversionVector=np.array(resourceConversionVector)

            currentState_futureResourceUtilisation=currentState_futureResourceUtilisation[resourceConversionVector]

            currentStateFuturnResourceUtilisation=currentState_futureResourceUtilisation.flatten()
            print('currentState_futureResourceUtilisation', currentStateFuturnResourceUtilisation)

            #----------------------------------------------------------------------------------------------------------------------------------------

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
            stateActionPossibilityPairsOfRun.append(currentStateActionPossibilityPairsOfRun)


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
            currentRunSimulation_output.stateActionPossibilityPairsOfBestRun = stateActionPossibilityPairsOfRun[indexBestRun]

    #print("end " + str(currentActivitySequence.fileName[:-4]))
    #print('-------------------------------------------------------------')


    return currentRunSimulation_output