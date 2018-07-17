import Queue as Q

import time

#import resource
import sys

import math
from heapq import heappush, heappop, heapify
## The Class that Represents the Puzzle
goal_state = (0,1,2,3,4,5,6,7,8)


class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []

        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i / self.n
                self.blank_col = i % self.n
                break

    def display(self):

        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print line

    def move_left(self):

        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:
            up_child = self.move_up()

            if up_child is not None:
                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:
                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:
                self.children.append(left_child)
            
            right_child = self.move_right()

            if right_child is not None:
                self.children.append(right_child)
        return self.children

# Function that Writes to output.txt
def writeOutput(state, expandedNodes):
    fp = open("output.txt", 'w')
    actions = []
    initial_state = state
    search_depth = 0
    while(state.parent != None):
        actions.append(state.action)
        state = state.parent
        search_depth += 1

    actions.reverse()
    fp.write("path to goal: ")

    for action in actions:
        fp.write(action)
        fp.write(" ")

    fp.write("\ncost of path:  " + str(initial_state.cost) + '\n')
    fp.write("nodes_expanded: " + str(expandedNodes) + '\n')
    fp.write("search_depth: " + str(search_depth) + '\n')
    fp.write("max_depth: " + str(search_depth + 1) + '\n')
   
def bfs_search(initial_state):
    frontier = Q.Queue()
    frontier.put(initial_state)
    explored = set()
    expandedNodes = 0

    while(not frontier.empty()):
        state = frontier.get()
        explored.add(state.config)
        
        if(state.config == goal_state):
            writeOutput(state, expandedNodes)
            return state

        children = state.expand()
        expandedNodes += 1
        
        for neighbor in children:
            if neighbor.config not in explored:
                frontier.put(neighbor) 
            
    return None
    

def dfs_search(initial_state):
    frontier = set()
    frontier.add(initial_state)
    explored = set()
    expandedNodes = 0

    while len(frontier) > 0:
        state = frontier.pop()
        explored.add(state.config)
        
        if(state.config == goal_state):
            writeOutput(state, expandedNodes)
            return state

        children = state.expand()
        expandedNodes += 1
        
        for neighbor in children:
            if (neighbor.config not in explored):
                frontier.add(neighbor) 
            
    return None

def A_star_search(initial_state):
    frontier = list()
    entry = (calculate_total_cost(initial_state), initial_state)
    heappush(frontier, entry)
    explored = set()
    expandedNodes = 0

    while len(frontier) > 0:
        state = heappop(frontier)[1]
        explored.add(state.config)
        if(state.config == goal_state):
            writeOutput(state, expandedNodes)
            return state

        children = state.expand()
        expandedNodes += 1
        
        for neighbor in children:
            if neighbor.config not in explored:
                entry = (calculate_total_cost(neighbor), neighbor)
                heappush(frontier, entry) 
                heapify(frontier)
                
    
    return None
    
def calculate_total_cost(state):
    sum = 0
    for i in range(1,len(state.config)):
        sum += calculate_manhattan_dist(i, state.config[i], state.n)
    return sum

def calculate_manhattan_dist(idx, value, n):
    start_row_index = idx / n
    start_col_index = idx % n 
    goal_row_index = value / n
    goal_col_index = idx % n
    return abs(start_row_index - goal_row_index) + abs(start_col_index - goal_col_index)

def test_goal(puzzle_state):
    if puzzle_state == [0,1,2,3,4,5,6,7,8]:
        return True
    return False


# Main Function that reads in Input and Runs corresponding Algorithm
def main():

    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)
    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:  
        print("Enter valid command arguments !")

if __name__ == '__main__':
    main()