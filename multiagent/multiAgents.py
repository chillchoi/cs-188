# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #start from the game score, then add bonus/penalty for this move
        #1 / (x + 1)
        moveScore = successorGameState.getScore()

        #this is for food: closer food = bigger bonus (+1 so we never divide by 0)
        foodPositions = newFood.asList()
        if foodPositions:
            nearestFoodDist = min([manhattanDistance(newPos,food) for food in foodPositions])
            moveScore += 1 / (nearestFoodDist + 1)

        #this is for ghosts: big penalty if a ghost is 1 step away or less
        nearestGhostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if nearestGhostDist <= 1:
            moveScore -= 500
        
        
        return moveScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #scoreState returns the minimax score of a state by calling itself on every possible next move
        def scoreState(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            #figure out whose turn is next, depth only goes up after the last ghost moves
            nextAgentIndex = agentIndex + 1
            nextDepthLevel = depth
            if nextAgentIndex == state.getNumAgents():
                nextAgentIndex = 0
                nextDepthLevel = depth + 1

            #pacman (max) picks the biggest score, ghosts (min) pick the smallest
            childScores = []
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                childScores.append(scoreState(nextState, nextDepthLevel, nextAgentIndex))

            if agentIndex == 0:
                return max(childScores)
            return min(childScores)

        #top level: try each pacman move and keep the one with the highest score
        chosenAction = None
        highestScore = float("-inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            moveScore = scoreState(nextState, 0, 1)
            if moveScore > highestScore:
                highestScore = moveScore
                chosenAction = action
        return chosenAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # returns the minimax score of a state, skipping branches that can't change the answer (alpha-beta)
        def scoreState(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            nextAgentIndex = agentIndex + 1
            nextDepthLevel = depth
            if nextAgentIndex == state.getNumAgents():
                nextAgentIndex = 0
                nextDepthLevel = depth + 1
                
            #this is for the pacman... (max) replace alpha if there is a bigger number
            #prune anything if the score is above beta
            if agentIndex == 0:
                bestSoFar = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    nextState = state.generateSuccessor(agentIndex, action)
                    bestSoFar = max(bestSoFar, scoreState(nextState, nextDepthLevel, nextAgentIndex, alpha, beta))
                    if bestSoFar > beta:
                        return bestSoFar
                    alpha = max(alpha, bestSoFar)
                return bestSoFar

            
            #this is for ghost: replace beta if there is a smaller score, 
            # prune anything if its below alpha
            bestSoFar = float("inf")
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                bestSoFar = min(bestSoFar, scoreState(nextState, nextDepthLevel, nextAgentIndex, alpha, beta))
                if bestSoFar < alpha:
                    return bestSoFar
                beta = min(beta, bestSoFar)
            return bestSoFar

        
        chosenAction = None
        highestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            moveScore = scoreState(nextState, 0, 1, alpha, beta)
            if moveScore > highestScore:
                highestScore = moveScore
                chosenAction = action
            alpha = max(alpha, highestScore)
        return chosenAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #q2: scoreState returns the minimax score of a state by calling itself on every possible next move
        def scoreState(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            #figure out whose turn is next, depth only goes up after the last ghost moves
            nextAgentIndex = agentIndex + 1
            nextDepthLevel = depth
            if nextAgentIndex == state.getNumAgents():
                nextAgentIndex = 0
                nextDepthLevel = depth + 1

            # q2:pacman (max) picks the biggest score, ghosts (min) pick the smallest
            childScores = []
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                childScores.append(scoreState(nextState, nextDepthLevel, nextAgentIndex))

            if agentIndex == 0:
                return max(childScores)
            # SAME CODE AS Q2 just replaced min to avg for ghosts
            return sum(childScores) / len(childScores)

        #q2: top level: try each pacman move and keep the one with the highest score
        chosenAction = None
        highestScore = float("-inf")
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            moveScore = scoreState(nextState, 0, 1)
            if moveScore > highestScore:
                highestScore = moveScore
                chosenAction = action
        return chosenAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Starts from the state's game score and adds a food bonus. 
    The bonus is 1 / (distance to nearest food + 1), so closer food is worth more, 
    and the +1 avoids dividing by zero. It ignores ghosts and pellets. 
    It still won 10 out of 10 on smallClassic.>
    """
    "*** YOUR CODE HERE ***"
    #same structure as Q1 intitially
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    #find closest food (closer, bigger bonus, same structure as q1 but newer variables)
    
    if foodList:
        nearestFood = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 1.0 / (nearestFood + 1) 

    return score

# Abbreviation
better = betterEvaluationFunction
