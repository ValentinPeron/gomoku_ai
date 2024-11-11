import math
import numpy as np
import time
import random
from functools import lru_cache

# Initialize Zobrist hashing table for hashing board states
ZOBRIST_TABLE = np.random.randint(1, 2**63 - 1, size=(20, 20, 3), dtype=np.int64)

class Minimax:
    def __init__(self, board, max_depth=3):
        self.board = board
        self.size = len(board)
        self.max_depth = max_depth
        self.ai_role = 1
        self.opponent_role = 2
        self.transposition_table = {}

    def zobrist_hash(self):
        hash_value = 0
        for x in range(self.size):
            for y in range(self.size):
                piece = self.board[x][y]
                if piece != 0:
                    hash_value ^= ZOBRIST_TABLE[x][y][piece]
        return hash_value

    # Check if there's a five-in-a-row for a given player
    def is_winning_move(self, role):
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == role:
                    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        if self.check_direction(x, y, role, (dx, dy)):
                            return True
        return False

    # Check for five-in-a-row in a particular direction
    def check_direction(self, x, y, role, direction):
        dx, dy = direction
        count = 0
        for _ in range(5):
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == role:
                count += 1
                if count == 5:
                    return True
            else:
                break
            x += dx
            y += dy
        return False

    # Generate available moves around the existing pieces
    def available_moves(self):
        taken_moves = [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] != 0]
        area_range = 2
        available_moves = set()

        for (x, y) in taken_moves:
            for dx in range(-area_range, area_range + 1):
                for dy in range(-area_range, area_range + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == 0:
                        available_moves.add((nx, ny))
        return list(available_moves)

    def minimax(self, depth, alpha, beta, maximizing_player):
        hash_value = self.zobrist_hash()
        if hash_value in self.transposition_table and self.transposition_table[hash_value]["depth"] >= depth:
            return self.transposition_table[hash_value]["score"], None

        if depth == 0 or self.is_winning_move(self.ai_role) or self.is_winning_move(self.opponent_role):
            score = self.evaluate_board()
            self.transposition_table[hash_value] = {"score": score, "depth": depth}
            return score, None

        best_move = None
        moves = self.available_moves()

        for move in moves:
            x, y = move
            self.board[x][y] = self.ai_role if maximizing_player else self.opponent_role

            eval, _ = self.minimax(depth - 1, alpha, beta, not maximizing_player)

            self.board[x][y] = 0  # Undo move

            if maximizing_player:
                if eval > alpha:
                    alpha = eval
                    best_move = move
            else:
                if eval < beta:
                    beta = eval
                    best_move = move

            if beta <= alpha:
                break

        return (alpha if maximizing_player else beta), best_move

    def evaluate_board(self):
        ai_score = self.evaluate_lines(self.ai_role)
        opponent_score = self.evaluate_lines(self.opponent_role)
        return ai_score * 1.2 - opponent_score

    def evaluate_lines(self, role):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        cached_scores = {}

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == role:
                    for dx, dy in directions:
                        if (x, y, dx, dy) not in cached_scores:
                            length = self.count_in_line(x, y, dx, dy, role)
                            cached_scores[(x, y, dx, dy)] = length
                        else:
                            length = cached_scores[(x, y, dx, dy)]

                        if length == 5:
                            return 10000
                        elif length == 4:
                            score += 100
                        elif length == 3:
                            score += 10
                        elif length == 2:
                            score += 1
        return score

    def count_in_line(self, x, y, dx, dy, role):
        count = 0
        for _ in range(5):
            if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == role:
                count += 1
            else:
                break
            x += dx
            y += dy
        return count


class Game:
    def __init__(self, size, board, depth, ai1, ai2):
        self.size = size
        self.board = board
        self.depth = depth
        self.ai1 = ai1(self.board, depth)
        self.ai2 = ai2(self.board, depth)
        self.ai1_role = 1
        self.ai2_role = 2
        self.current_role = self.ai1_role
        self.round_number = 1
        self.first_move_initialised = False

    def print_board_state(self, available_moves_ai):
        BLUE = '\033[94m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        EMPTY = '·'
        PLAYER1 = '○'  # AI 1
        PLAYER2 = '●'  # AI 2

        print(f"\n{BOLD}Round {self.round_number}: AI {self.current_role}'s turn{RESET}")
        print('    ', end='')
        for i in range(self.size):
            print(f"{i:2}", end=' ')
        print('\n    ' + '─' * (self.size * 3))

        for i, row in enumerate(self.board):
            print(f"{i:2} │", end=' ')
            for j, cell in enumerate(row):
                if cell == 0:
                    if (i, j) in available_moves_ai:
                        print(f"{YELLOW}{EMPTY}{RESET}", end=' ')
                    else:
                        print(EMPTY, end=' ')
                elif cell == 1:
                    print(f"{BLUE}{PLAYER1}{RESET}", end=' ')
                else:
                    print(f"{RED}{PLAYER2}{RESET}", end=' ')
            print(f"│ {i:2}")

        print('    ' + '─' * (self.size * 3))
        print('    ', end='')
        for i in range(self.size):
            print(f"{i:2}", end=' ')
        print('\n')

    def display_board(self, start_time, best_move, available_moves_ai):
        print(f"Think time: {time.time() - start_time:.2f}s")
        x, y = best_move
        if self.board[x][y] == 0:
            self.board[x][y] = self.current_role
        else:
            raise ValueError("Error move not possible")

        self.print_board_state(available_moves_ai)

    def play(self):
        start_game = time.time()
        while True:
            print(f"Round {self.round_number}: AI {self.current_role} is making a move...")

            start = time.time()
            if not self.first_move_initialised:
                self.first_move_initialised = True
                best_move = (self.size // 2, self.size // 2)
            elif self.current_role == self.ai1_role:
                _, best_move = self.ai1.minimax(self.depth, -math.inf, math.inf, True)
            else:
                _, best_move = self.ai2.minimax(self.depth, -math.inf, math.inf, True)

            available_moves_ai = self.ai1.available_moves()
            self.display_board(start, best_move, available_moves_ai)

            if self.ai1.is_winning_move(self.current_role):
                print(f"AI {self.current_role} wins!")
                break

            self.current_role = self.ai2_role if self.current_role == self.ai1_role else self.ai1_role
            self.round_number += 1
        print(f"Game took : {time.time() - start_game:.2f}s | Avg time per round : {(time.time() - start_game) / self.round_number:.2f}s")

def main():
    size = 20
    board = [[0 for _ in range(size)] for _ in range(size)]
    game = Game(size, board, 3, Minimax, Minimax)
    game.play()

if __name__ == "__main__":
    main()

