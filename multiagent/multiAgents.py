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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        ghost_positions = successorGameState.getGhostPositions()
        min_food_dis = float('inf')
        for food in newFood.asList():
            min_food_dis = min(min_food_dis, manhattanDistance(food, newPos))
        min_ghost_dis = float('inf')
        for ghost_position in ghost_positions:
            min_ghost_dis = min(min_ghost_dis, manhattanDistance(ghost_position, newPos))
        # print([min_food_dis, min_dis, successorGameState.getNumFood()])
        if min_ghost_dis == 0:
            return -float('inf')
        return 1 / min_food_dis - 5 / min_ghost_dis - successorGameState.getNumFood()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        MOD = gameState.getNumAgents()
        depth = self.depth

        def value(state: GameState, agent_index, cur_depth):
            if state.isWin() or state.isLose() or cur_depth > depth:
                return self.evaluationFunction(state)
            if agent_index == 0:
                return max_value(state, agent_index, cur_depth)
            return min_value(state, agent_index, cur_depth)

        def max_value(state: GameState, agent_index, cur_depth):
            res = -float('inf')
            actions = state.getLegalActions(agent_index)
            successors = [state.generateSuccessor(agent_index, action) for action in actions]
            for successor in successors:
                res = max(value(successor, (agent_index + 1) % MOD, cur_depth), res)
            return res

        def min_value(state: GameState, agent_index, cur_depth):
            res = float('inf')
            actions = state.getLegalActions(agent_index)
            successors = [state.generateSuccessor(agent_index, action) for action in actions]
            if agent_index == MOD - 1:
                cur_depth += 1
            for successor in successors:
                res = min(value(successor, (agent_index + 1) % MOD, cur_depth), res)
            return res

        actions = gameState.getLegalActions(0)
        opt_action = actions[0]
        res = -float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if res < value(successor, 1, 1):
                opt_action = action
                res = value(successor, 1, 1)
        return opt_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        MOD = gameState.getNumAgents()
        depth = self.depth

        def value(state: GameState, agent_index, cur_depth, a, b):
            if state.isWin() or state.isLose() or cur_depth > depth:
                return self.evaluationFunction(state)
            if agent_index == 0:
                return max_value(state, agent_index, cur_depth, a, b)
            return min_value(state, agent_index, cur_depth, a, b)

        def max_value(state: GameState, agent_index, cur_depth, a, b):
            res = -float('inf')
            actions = state.getLegalActions(agent_index)
            for action in actions:
                successor = state.generateSuccessor(agent_index, action)
                res = max(value(successor, (agent_index + 1) % MOD, cur_depth, a, b), res)
                if res > b:
                    return res
                a = max(a, res)
            return res

        def min_value(state: GameState, agent_index, cur_depth, a, b):
            res = float('inf')
            actions = state.getLegalActions(agent_index)
            if agent_index == MOD - 1:
                cur_depth += 1
            for action in actions:
                successor = state.generateSuccessor(agent_index, action)
                res = min(value(successor, (agent_index + 1) % MOD, cur_depth, a, b), res)
                if res < a:
                    return res
                b = min(b, res)
            return res

        actions = gameState.getLegalActions(0)
        opt_action = actions[0]
        res = -float('inf')
        a = -float('inf')
        b = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if res < value(successor, 1, 1, a, b):
                opt_action = action
                res = value(successor, 1, 1, a, b)
                a = max(a, res)
        return opt_action


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
        MOD = gameState.getNumAgents()
        depth = self.depth

        def value(state: GameState, agent_index, cur_depth):
            if state.isWin() or state.isLose() or cur_depth > depth:
                return self.evaluationFunction(state)
            if agent_index == 0:
                return max_value(state, agent_index, cur_depth)
            return exp_value(state, agent_index, cur_depth)

        def max_value(state: GameState, agent_index, cur_depth):
            res = -float('inf')
            actions = state.getLegalActions(agent_index)
            successors = [state.generateSuccessor(agent_index, action) for action in actions]
            for successor in successors:
                res = max(value(successor, (agent_index + 1) % MOD, cur_depth), res)
            return res

        def exp_value(state: GameState, agent_index, cur_depth):
            res = 0
            actions = state.getLegalActions(agent_index)
            successors = [state.generateSuccessor(agent_index, action) for action in actions]
            if agent_index == MOD - 1:
                cur_depth += 1
            for successor in successors:
                res += value(successor, (agent_index + 1) % MOD, cur_depth)
            return res / len(actions)

        actions = gameState.getLegalActions(0)
        opt_action = actions[0]
        res = -float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if res < value(successor, 1, 1):
                opt_action = action
                res = value(successor, 1, 1)
        return opt_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    ghost_positions = currentGameState.getGhostPositions()
    total_dis = 0
    min_food_dis = float('inf')
    for food in newFood.asList():
        min_food_dis = min(min_food_dis, manhattanDistance(food, newPos))
        total_dis += manhattanDistance(food, newPos)
    min_ghost_dis = float('inf')
    for ghost_position in ghost_positions:
        min_ghost_dis = min(min_ghost_dis, manhattanDistance(ghost_position, newPos))
    if min_ghost_dis <= 3:
        return -float('inf')
    if currentGameState.getNumFood() == 0:
        return float('inf')
    num_capsules = len(currentGameState.getCapsules())
    # print([min_food_dis, min_ghost_dis, currentGameState.getNumFood(), len(currentGameState.getCapsules())])
    return 20.0 / min_food_dis - 10.0 / min_ghost_dis - 30 * currentGameState.getNumFood() - 100 * num_capsules


# Abbreviation
better = betterEvaluationFunction
