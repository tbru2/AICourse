from random import randint

from BaseAI import BaseAI

import math

class PlayerAI(BaseAI):
    def getMove(self, grid):
        bestMove = ""
        maxUtility = -float('inf')
        moves = grid.getAvailableMoves()
        
        for move in moves:
            childGrid = grid.clone()
            childGrid.move(move)
            _, utility = self.maximize(childGrid, -float('inf'), float('inf'), 0)
            if utility > maxUtility:
                maxUtility = utility
                bestMove = move
        return bestMove 

    def maximize(self, grid, alpha, beta, depth ):
    
        if depth >= 3:
            return None, maxUtility

        maxChild, maxUtility = None, -float('inf')

        moves = grid.getAvailableMoves()

        for move in moves:
            childGrid = grid.clone()
            childGrid.move(move)

            utility = self.minimize(childGrid, alpha, beta, depth + 1)[1]
            if utility > maxUtility:
                maxChild, maxUtility, bestMove = childGrid, utility, move
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility

    def minimize(self, grid, alpha, beta, depth):
        
        if depth >= 3:
            return [grid, self.sumGrid(grid), None]

        minChild, minUtility = None, float('inf')

        moves = grid.getAvailableMoves()

        for move in moves:
            childGrid = grid.clone()
            childGrid.move(move)
            utility = self.maximize(childGrid, alpha, beta, depth + 1)[1]

            if utility < minUtility:
                minChild, minUtility, bestMove = childGrid, utility, move
            if minUtility <= alpha:
                break
            if minUtility <= beta:
                beta = minUtility

        return minChild, minUtility

    def sumGrid(self, grid):
        sum = grid.getMaxTile()

        for x in range(grid.size):
            for y in range(grid.size):
                if grid.getCellValue([x,y]) === 0:
                    sum += 1
        return sum 
    