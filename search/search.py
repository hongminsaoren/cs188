# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    closed = set()
    q = util.Stack()
    start_state = problem.getStartState()
    q.push([start_state, []])
    while not q.isEmpty():
        cur_node = q.pop()
        if problem.isGoalState(cur_node[0]):
            return cur_node[1]
        if cur_node[0] not in closed:
            closed.add(cur_node[0])
            for child_node in problem.getSuccessors(cur_node[0]):
                cpy = cur_node[1].copy()
                cpy.append(child_node[1])
                q.push([child_node[0], cpy])


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """
    s.push([start_state, []])
    visited = []
    visited.append(start_state)
    while not s.isEmpty():
        cur_state = s.pop()
        path = cur_state[1]
        if problem.isGoalState(cur_state[0]):
            return path
        for next_state in problem.getSuccessors(cur_state[0]):
            if next_state[0] not in visited:
                visited.append(next_state[0])
                tmp = path.copy()
                tmp.append(next_state[1])
                s.push([next_state[0], tmp])
    """
    closed = set()
    q = util.Queue()
    start_state = problem.getStartState()
    q.push([start_state, []])
    while not q.isEmpty():
        cur_node = q.pop()
        if problem.isGoalState(cur_node[0]):
            return cur_node[1]
        if cur_node[0] not in closed:
            closed.add(cur_node[0])
            for child_node in problem.getSuccessors(cur_node[0]):
                cpy = cur_node[1].copy()
                cpy.append(child_node[1])
                q.push([child_node[0], cpy])


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """
    from util import PriorityQueue
    q = PriorityQueue()
    start_state = problem.getStartState()
    print(start_state)
    state_to_path = {start_state: [[], 0]}
    q.push(start_state, 0)
    visited = []
    while not q.isEmpty():
        cur_state = q.pop()
        visited.append(cur_state)
        path = state_to_path[cur_state][0]
        if problem.isGoalState(cur_state):
            return path
        successors = problem.getSuccessors(cur_state)
        for next_state in successors:
            if next_state[0] not in visited:
                tmp = path.copy()
                tmp.append(next_state[1])
                priority = problem.getCostOfActions(tmp)
                if next_state[0] in state_to_path:
                    if priority < state_to_path[next_state[0]][1]:
                        state_to_path[next_state[0]] = [tmp, priority]
                q.update(next_state[0], priority)
"""
    closed = set()
    q = util.PriorityQueue()
    start_state = problem.getStartState()
    q.push([start_state, []], 0)
    while not q.isEmpty():
        cur_node = q.pop()
        if problem.isGoalState(cur_node[0]):
            return cur_node[1]
        if cur_node[0] not in closed:
            closed.add(cur_node[0])
            for child_node in problem.getSuccessors(cur_node[0]):
                cpy = cur_node[1].copy()
                cpy.append(child_node[1])
                priority = problem.getCostOfActions(cpy)
                q.update([child_node[0], cpy], priority)
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    q = util.PriorityQueue()
    start_state = problem.getStartState()
    q.push([start_state, []], heuristic(start_state, problem))
    while not q.isEmpty():
        cur_node = q.pop()
        if problem.isGoalState(cur_node[0]):
            return cur_node[1]
        if cur_node[0] not in closed:
            closed.add(cur_node[0])
            for child_node in problem.getSuccessors(cur_node[0]):
                cpy = cur_node[1].copy()
                cpy.append(child_node[1])
                priority = problem.getCostOfActions(cpy) + heuristic(child_node[0], problem)
                q.update([child_node[0], cpy], priority)
    return None
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
