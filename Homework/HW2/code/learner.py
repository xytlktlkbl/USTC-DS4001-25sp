from submission import *

if __name__ == "__main__":
    chess_board = Gobang(board_size=4, bound=3)
    chess_board.learning(episodes=10000)
