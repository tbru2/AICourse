from copy import deepcopy
import sys

ROWS = list("ABCDEFGHI")
COLUMNS = list("123456789")

class Sudoku:

    def __init__(self, boardArr):
        n =0
        self.board = {}
        self.domains = {}
        for letter in ROWS:
            for column in COLUMNS:
                self.board[letter + column] = int(boardArr[n])
                if int(boardArr[n]) != 0:
                    self.domains[letter + column] = [int(boardArr[n])]
                else:
                    self.domains[letter + column] = [i for i in range(1,10)]
                n+=1
        n = 0
        
        self.constraints = dict()
        i = 0
        boxes = list()
        columnsIndex = 0
        rowsIndex = 0
        
        for f in range(1,10):
            box = []
            for m in range(3):
                for n in range(3):
                    box.append(ROWS[m+rowsIndex] + COLUMNS[n+columnsIndex])
            boxes.append(box)
            columnsIndex += 3
            if f % 3 == 0:
                rowsIndex += 3
                columnsIndex = 0
    
        for key in self.domains.keys():
            letter, column = list(key)
            sameRow = list(letter + number for number in COLUMNS)
            sameColumn = list(a + column for a in ROWS)
            sameBox = list()
            
            for box in boxes:
                if key in box:
                    sameBox = box
            
            self.constraints[key] = set(sameRow + sameColumn + sameBox)
                 
    def AC3(self):
        queue = []
        for key in self.constraints:
            for neighbor in self.constraints[key]:
                 queue.append((key,neighbor))

        while len(queue) > 0:
            xi, xj = queue.pop()
            if self.revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                for neighbor in self.constraints[xi]:
                    queue.append((neighbor,xi))
    
        return True

    def revise(self,xi,xj):
        if self.board[xi] != self.board[xj] and self.board[xj] != 0:
            if self.board[xj] in self.domains[xi]:
                self.domains[xi].remove(self.board[xj])
                if len(self.domains[xi]) == 1:
                    self.board[xi] = self.domains[xi][0]
                return True
        return False
    
    
    def bts(self):
        return self.btsHelper(deepcopy(self.board), deepcopy(self.domains))

    def btsHelper(self, board, domains):
        if self.isComplete():
            return True

        key = self.select_unassigned_variable()
        for value in domains[key]:
            if self.isConsistent(value, key):
                self.board[key] = value
                self.domains[key] = [value]
                isSuccessful= self.forwardCheck(key)
                if isSuccessful:
                    result = self.btsHelper(deepcopy(self.board), deepcopy(self.domains))
                    if result == True:
                        return result
                    self.board = deepcopy(board)
                    self.domains = deepcopy(domains)
        return False

    def forwardCheck(self,key):
        for neighbor in self.constraints[key]:
            if self.revise(key, neighbor):
                if len(self.domains[key]) == 0 or len(self.domains[neighbor]) == 0:# or not self.isConsistent(self.board[key],key):
                    return False
        return True

    def isConsistent(self, value, key):
        for constraint in self.constraints[key]:
            if value == self.board[constraint]:
                return False
        return True

    def select_unassigned_variable(self):
        minIndex = 0
        minValue = float('inf')
        for key in self.board:
            if len(self.domains[key]) < minValue and self.board[key] == 0:
                minIndex = key
                minValue = len(self.domains[key])
        return minIndex

    def isComplete(self):
        for key in self.board:
            if self.board[key] == 0:
                return False
        return True

    def display(self):
        k = 0
        print '- - - - - - - - - - - - - - - - - -'
        for i in range(len(ROWS)):
            print("|"),
            for j in range(len(COLUMNS)):
                print (str(self.board[ROWS[i] + COLUMNS[j]]) + " "), 
                if (j+1) % 3 == 0:
                    print ("|"),
                k+=1
            print ""
            if (i+1) % 3 == 0:
                print ('- - - - - - - - - - - - - - - - - -')

    def outputToFile(self, fp, searchType):
        for i in range(len(ROWS)):
            for j in range(len(COLUMNS)):
                fp.write(str(self.board[ROWS[i] + COLUMNS[j]]))
        fp.write(" " + searchType + '\n')

sudokuBoard = Sudoku(list(sys.argv[1]))
sudokuBoard.display()
sudokuBoard.AC3()
searchType = "AC3"
if not sudokuBoard.isComplete():
    sudokuBoard.bts()
    searchType = "BTS"
sudokuBoard.display()
fp = open("output.txt", 'w')
sudokuBoard.outputToFile(fp, searchType)
