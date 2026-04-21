import time

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "-":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "-":
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2] != "-":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "-":
        return board[0][2]

    return None

def is_full(board):
    for row in board:
        if "-" in row:
            return False
    return True

def find_best_move(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == "-":
                return (i, j)  # first empty cell

while True:
    board = [["-" for _ in range(3)] for _ in range(3)]
    total_moves = 0
    turn_number = 1
    ai_total_time = 0
    ai_moves = 0

    print("Who plays first?")
    print("1. Human (O)")
    print("2. AI (X)")
    choice = input("Enter 1 or 2: ")
    human_turn = True if choice == "1" else False

    while True:
        print_board(board)
        print(f"Turn {turn_number}")

        if human_turn:
            start_time = time.time()
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter col (0-2): "))

                    if row not in range(3) or col not in range(3):
                        print("Invalid index!")
                        continue
                    if board[row][col] != "-":
                        print("Cell already filled!")
                        continue
                    break
                except:
                    print("Enter numbers only!")

            print(f"Human move time: {time.time() - start_time:.4f} sec")
            board[row][col] = "O"
            total_moves += 1

        else:
            print("AI is thinking...")
            start_time = time.time()

            move = find_best_move(board)

            ai_time = time.time() - start_time
            ai_total_time += ai_time
            ai_moves += 1

            print(f"AI decision time: {ai_time:.4f} sec")

            board[move[0]][move[1]] = "X"
            total_moves += 1

        winner = check_winner(board)
        if winner:
            print_board(board)
            print("Winner:", winner)
            break

        if is_full(board):
            print_board(board)
            print("Draw!")
            break

        human_turn = not human_turn
        turn_number += 1

    if ai_moves > 0:
        print(f"Avg AI time: {ai_total_time / ai_moves:.4f} sec")

    replay = input("Play again? (y/n): ")
    if replay.lower() != "y":
        break
