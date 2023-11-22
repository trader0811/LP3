#!/usr/bin/env python
# coding: utf-8

# In[35]:


#DAA-1 Fibonacci using Recursive and Non Recursive
# Non-recursive program
def nonrec(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    f0=0
    f1=1
    for _ in range(2,n+1):
        f2=f0+f1
        f0=f1
        f1=f2
    return f2

n=10
print("Non Recursive:")
for i in range(n):
    print(nonrec(i))
    
    
# Recursive program
def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

print("Recursive:")
for i in range(10):
    print(fib(i))



# In[41]:


#DAA-2 Huffman Encoding using greedy strategy method
import heapq 

class node: 
	def __init__(self, freq, symbol, left=None, right=None): 
		self.freq = freq 
		self.symbol = symbol 
		self.left = left  
		self.right = right 
		self.huff = '' 

	def __lt__(self, nxt): 
		return self.freq < nxt.freq 

def printNodes(node, val=''): 
	newVal = val + str(node.huff) 

	if(node.left): 
		printNodes(node.left, newVal) 
	if(node.right): 
		printNodes(node.right, newVal) 

	if(not node.left and not node.right): 
		print(f"{node.symbol} -> {newVal}") 

chars = ['a', 'b', 'c', 'd', 'e', 'f'] 
# freq = [5, 9, 12, 13, 16, 45] 
freq = [50,10,30,5,3,2] 

nodes = [] 
for x in range(len(chars)): 
	heapq.heappush(nodes, node(freq[x], chars[x])) 

while len(nodes) > 1: 
 
	left = heapq.heappop(nodes) 
	right = heapq.heappop(nodes) 
	left.huff = 0
	right.huff = 1
	newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right) 
	heapq.heappush(nodes, newNode) 

printNodes(nodes[0]) 


# In[45]:


#DAA-3 fractional Knapsack problem using a greedy method
class Item:
	def __init__(self, profit, weight):
		self.profit = profit
		self.weight = weight

def fractionalKnapsack(W, arr):
	arr.sort(key=lambda x: (x.profit/x.weight), reverse=True) 
	finalvalue = 0.0
	for item in arr:
		if item.weight <= W:
			W -= item.weight
			finalvalue += item.profit
		else:
			finalvalue += item.profit * W / item.weight
			break
	return finalvalue

# Driver Code
if __name__ == "__main__":
	W = 50
	arr = [Item(60, 10), Item(100, 20), Item(120, 30)]
	max_val = fractionalKnapsack(W, arr)
	print(max_val)


# In[57]:


#DAA-4 0/1 Knapsack Problem
def solve_knapsack():
    val = [50, 100, 150, 200]  # value array
    wt = [8, 16, 32, 40]  # Weight array
    W = 64
    n = len(val) - 1

    def knapsack(W, n):  # (Remaining Weight, Number of items checked)
        # base case
        if n < 0 or W <= 0:
            return 0
        # Higher weight than available
        if wt[n] > W:
            return knapsack(W, n - 1)
        else:
            return max(val[n] + knapsack(W - wt[n], n - 1), knapsack(W, n - 1))
        # max(including , not including)

    print(knapsack(W, n))

if __name__ == "__main__":
    solve_knapsack()


# In[11]:


#DAA-5 N QUEENS
def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False    
    return True
def solve_nqueens(board, row, n):
    if row == n:
        return True    
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            if solve_nqueens(board, row + 1, n):
                return True    
            board[row][col] = 0
    return False

def print_board(board):
    for row in board:
        print(" ".join("Q" if cell == 1 else "." for cell in row))

def nqueens(n):
    board = [[0] * n for _ in range(n)]
    board[0][0] = 1
    if not solve_nqueens(board, 1, n):
        print("No solution exists.")
    else:
        print_board(board)
nqueens(8)


# In[12]:


def is_safe(board, row, col, n):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 1:
            return False
    
    # Check upper left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    
    # Check upper right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False
    
    return True

def solve_nqueens(board, row, n):
    if row == n:
        # All queens are placed successfully
        return True
    
    for col in range(n):
        if is_safe(board, row, col, n):
            # Place the queen
            board[row][col] = 1

            # Recur to place the remaining queens
            if solve_nqueens(board, row + 1, n):
                return True  # If a solution is found, stop the recursion
            
            # If placing the queen in the current position doesn't lead to a solution,
            # backtrack and try the next column
            board[row][col] = 0

    # If no column is suitable, return False to trigger backtracking
    return False

def print_board(board):
    for row in board:
        print(" ".join("Q" if cell == 1 else "." for cell in row))

def nqueens(n):
    # Initialize an empty chessboard
    board = [[0] * n for _ in range(n)]

    # Place the first queen in the first row
    board[0][0] = 1

    # Solve the N-Queens problem using backtracking
    if not solve_nqueens(board, 1, n):
        print("No solution exists.")
    else:
        print_board(board)

# Example usage:
nqueens(8)


# In[13]:


def n_queens(n):
    col = set()
    posDiag = set()  # (r+c)
    negDiag = set()  # (r-c)
    res = []
    board = [["0"] * n for i in range(n)]

    def backtrack(r):
        if r == n:
            copy = [" ".join(row) for row in board]
            res.append(copy)
            return

        for c in range(n):
            if c in col or (r+c) in posDiag or (r-c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r+c)
            negDiag.add(r-c)
            board[r][c] = "1"

            backtrack(r+1)

            col.remove(c)
            posDiag.remove(r+c)
            negDiag.remove(r-c)
            board[r][c] = "0"

    backtrack(0)  # Call the backtrack function with the initial row value

    for sol in res:
        for row in sol:
            print(row)
        print()

if __name__ == "__main__":
    n_queens(8)

