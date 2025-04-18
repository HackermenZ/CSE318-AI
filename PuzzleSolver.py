import math
import heapq

# PuzzleState class to represent a state of the puzzle
class PuzzleState:
    def __init__(self, board, blank_pos, parent=None, g=0, h=0, heuristic_type="manhattan"):
        self.board = board
        self.blank_pos = blank_pos
        self.parent = parent
        self.g = g  # Cost from start
        self.h = h  # Heuristic value
        self.f = g + h  # Total cost (f = g + h)
        self.heuristic_type = heuristic_type

    def __eq__(self, other):
        return self.board == other.board  # Check if two boards are identical

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))  # To store the state in sets

    def __lt__(self, other):
        return self.f < other.f  # Compare by f value

    def is_goal(self, goal_state):
        return self.board == goal_state  # Check if current state matches the goal state

    def get_neighbors(self):
        neighbors = []
        row, col = self.blank_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        # For each direction, try to move the blank tile
        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < len(self.board) and 0 <= new_col < len(self.board[0]):
                new_board = [list(row) for row in self.board]  # Copy the current board
                new_board[row][col], new_board[new_row][new_col] = new_board[new_row][new_col], new_board[row][col]
                neighbors.append(PuzzleState(new_board, (new_row, new_col), self, self.g + 1, self.calculate_heuristic(new_board)))

        return neighbors

    def calculate_heuristic(self, board):
        if self.heuristic_type == "manhattan":
            return self.calculate_manhattan(board)
        elif self.heuristic_type == "hamming":
            return self.calculate_hamming(board)
        elif self.heuristic_type == "euclidean":
            return self.calculate_euclidean(board)
        elif self.heuristic_type == "linear_conflict":
            return self.calculate_linear_conflict(board)
        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

    def calculate_manhattan(self, board):
        total_distance = 0
        goal_positions = {i: (i // len(board), i % len(board)) for i in range(1, len(board) ** 2)}  # Goal positions for tiles 1 to n-1
        goal_positions[0] = (len(board) - 1, len(board) - 1)  # Goal position for the blank (0)

        for r in range(len(board)):
            for c in range(len(board[0])):
                tile = board[r][c]
                if tile != 0:  # Ignore the blank tile
                    goal_r, goal_c = goal_positions[tile]
                    total_distance += abs(r - goal_r) + abs(c - goal_c)

        return total_distance

    def calculate_hamming(self, board):
        total_misplaced = 0
        goal_positions = {i: i for i in range(1, len(board) ** 2)}
        goal_positions[0] = 0  # Goal position for the blank (0)

        for r in range(len(board)):
            for c in range(len(board[0])):
                tile = board[r][c]
                if tile != 0 and tile != goal_positions[tile]:
                    total_misplaced += 1

        return total_misplaced

    def calculate_euclidean(self, board):
        total_distance = 0
        goal_positions = {i: (i // len(board), i % len(board)) for i in range(1, len(board) ** 2)}  # Goal positions for tiles 1 to n-1
        goal_positions[0] = (len(board) - 1, len(board) - 1)  # Goal position for the blank (0)

        for r in range(len(board)):
            for c in range(len(board[0])):
                tile = board[r][c]
                if tile != 0:  # Ignore the blank tile
                    goal_r, goal_c = goal_positions[tile]
                    total_distance += math.sqrt((r - goal_r) ** 2 + (c - goal_c) ** 2)

        return total_distance

    def calculate_linear_conflict(self, board):
        manhattan_distance = self.calculate_manhattan(board)
        linear_conflicts = 0

        for i in range(len(board)):
            for j in range(len(board[0])):
                for k in range(j + 1, len(board[0])):
                    if board[i][j] != 0 and board[i][k] != 0:
                        if self.is_in_conflict(board[i][j], board[i][k], i, j, i, k):
                            linear_conflicts += 1

        return manhattan_distance + 2 * linear_conflicts

    def is_in_conflict(self, tile1, tile2, r1, c1, r2, c2):
        goal_positions = {i: (i // len(self.board), i % len(self.board)) for i in range(1, len(self.board) ** 2)}
        goal_positions[0] = (len(self.board) - 1, len(self.board) - 1)  # Goal position for the blank (0)

        goal_r1, goal_c1 = goal_positions[tile1]
        goal_r2, goal_c2 = goal_positions[tile2]

        if (goal_r1 == goal_r2 and goal_c1 > goal_c2) or (goal_c1 == goal_c2 and goal_r1 > goal_r2):
            return True
        return False


class PuzzleSolver:
    def __init__(self, start_board, goal_state, heuristic_type="manhattan"):
        self.start_state = PuzzleState(start_board, self.find_blank_position(start_board), h=0, heuristic_type=heuristic_type)
        self.goal_state = goal_state
        self.open_list = []  # Priority queue (min-heap) for A* search
        self.closed_list = set()  # Set to keep track of visited states
        self.path = []  # List to store the sequence of board configurations
        heapq.heappush(self.open_list, (self.start_state.f, self.start_state))  # Push initial state into the queue

    def find_blank_position(self, board):
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 0:
                    return r, c
        return None

    def solve(self):
        while self.open_list:
            _, current_state = heapq.heappop(self.open_list)  # Pop the state with the lowest f-value
            self.closed_list.add(current_state)  # Add the current state to the closed list

            if current_state.is_goal(self.goal_state):
                self.reconstruct_path(current_state)
                return self.path

            for neighbor in current_state.get_neighbors():
                if neighbor not in self.closed_list:
                    heapq.heappush(self.open_list, (neighbor.f, neighbor))  # Push neighbor with its f value

        return None  # No solution found

    def reconstruct_path(self, state):
        while state:
            self.path.append(state.board)
            state = state.parent
        self.path.reverse()  # Reverse the path to get the solution from start to goal


def get_user_input():
    k = int(input("Enter grid size N (e.g., for a 3x3 grid, enter 3): "))
    board = []
    print(f"Enter the initial {k}x{k} board configuration, row by row, with 0 for the blank tile:")
    for i in range(k):
        row = list(map(int, input().split()))
        board.append(row)

    goal_state = [[(i * k + j + 1) % (k * k) for j in range(k)] for i in range(k)]
    return board, goal_state


def is_solvable(board):
    flat_board = [tile for row in board for tile in row]
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] != 0 and flat_board[j] != 0 and flat_board[i] > flat_board[j]:
                inversions += 1
    return inversions % 2 == 0


def main():
    start_board, goal_state = get_user_input()

    if not is_solvable(start_board):
        print("The puzzle is unsolvable!")
    else:
        best_solution = None
        best_heuristic = None
        min_moves = float('inf')

        heuristics = ["manhattan", "hamming", "euclidean", "linear_conflict"]
        for heuristic in heuristics:
            print(f"\nSolving with {heuristic} heuristic:")
            solver = PuzzleSolver(start_board, goal_state, heuristic_type=heuristic)
            solution_path = solver.solve()

            if solution_path:
                print(f"Solution found with {heuristic} heuristic:")
                print(f"Number of moves: {len(solution_path) - 1}")
                print(f"Nodes expanded: {len(solver.closed_list)}")

                if len(solution_path) - 1 < min_moves:
                    min_moves = len(solution_path) - 1
                    best_solution = solution_path
                    best_heuristic = heuristic

        if best_solution:
            print(f"\nBest solution found using {best_heuristic} heuristic:")
            print(f"Minimum number of moves: {min_moves}")
            print(f"Solution path:")
            for step in best_solution:
                for row in step:
                    print(row)
                print()
        else:
            print("No solution found")


if __name__ == "__main__":
    main()
