package tictactoe_example

import "core:fmt"

EMPTY :: 0
PLAYER_X :: 1
PLAYER_O :: -1
BOARD_SIZE :: 9
UNIQUE_BOARD_STATES :: 19_683
MINIMAX_CACHE_SIZE :: UNIQUE_BOARD_STATES * 2

WIN_LINES: [8][3]int = {
	{0, 1, 2},
	{3, 4, 5},
	{6, 7, 8},
	{0, 3, 6},
	{1, 4, 7},
	{2, 5, 8},
	{0, 4, 8},
	{2, 4, 6},
}

Minimax_Cache :: struct {
	known:  [MINIMAX_CACHE_SIZE]bool,
	scores: [MINIMAX_CACHE_SIZE]int,
}

Dataset :: struct {
	inputs: [dynamic][BOARD_SIZE]f32,
	labels: [dynamic][BOARD_SIZE]f32,
}

destroy_dataset :: proc(dataset: ^Dataset) {
	delete(dataset.inputs)
	delete(dataset.labels)
	dataset^ = {}
}

make_empty_board :: proc() -> [BOARD_SIZE]int {
	return [BOARD_SIZE]int{}
}

other_player :: proc(player: int) -> int {
	if player == PLAYER_X {
		return PLAYER_O
	}
	return PLAYER_X
}

player_name :: proc(player: int) -> string {
	if player == PLAYER_X {
		return "X"
	}
	return "O"
}

empty_cell_label :: proc(index: int) -> string {
	switch index {
	case 0:
		return "1"
	case 1:
		return "2"
	case 2:
		return "3"
	case 3:
		return "4"
	case 4:
		return "5"
	case 5:
		return "6"
	case 6:
		return "7"
	case 7:
		return "8"
	case 8:
		return "9"
	case:
		return "?"
	}
}

cell_symbol :: proc(board: [BOARD_SIZE]int, index: int) -> string {
	switch board[index] {
	case PLAYER_X:
		return "X"
	case PLAYER_O:
		return "O"
	case:
		return empty_cell_label(index)
	}
}

print_board :: proc(board: [BOARD_SIZE]int) {
	for row := 0; row < 3; row += 1 {
		base := row * 3
		fmt.printf(
			" %s | %s | %s\n",
			cell_symbol(board, base + 0),
			cell_symbol(board, base + 1),
			cell_symbol(board, base + 2),
		)
		if row < 2 {
			fmt.println("---+---+---")
		}
	}
}

is_legal_move :: proc(board: [BOARD_SIZE]int, move: int) -> bool {
	if move < 0 || move >= BOARD_SIZE {
		return false
	}
	return board[move] == EMPTY
}

apply_move :: proc(board: ^[BOARD_SIZE]int, move: int, player: int) -> bool {
	if !is_legal_move(board^, move) {
		return false
	}
	board[move] = player
	return true
}

check_winner :: proc(board: [BOARD_SIZE]int) -> int {
	for line := 0; line < len(WIN_LINES); line += 1 {
		current_line := WIN_LINES[line]
		a := current_line[0]
		b := current_line[1]
		c := current_line[2]
		if board[a] != EMPTY && board[a] == board[b] && board[b] == board[c] {
			return board[a]
		}
	}

	return EMPTY
}

board_is_full :: proc(board: [BOARD_SIZE]int) -> bool {
	for i := 0; i < BOARD_SIZE; i += 1 {
		if board[i] == EMPTY {
			return false
		}
	}
	return true
}

is_terminal_board :: proc(board: [BOARD_SIZE]int) -> bool {
	return check_winner(board) != EMPTY || board_is_full(board)
}

board_to_input :: proc(board: [BOARD_SIZE]int, perspective: int) -> (input: [BOARD_SIZE]f32) {
	for i := 0; i < BOARD_SIZE; i += 1 {
		input[i] = f32(board[i] * perspective)
	}
	return
}

board_key :: proc(board: [BOARD_SIZE]int) -> int {
	key := 0
	factor := 1

	for i := 0; i < BOARD_SIZE; i += 1 {
		digit := 0
		switch board[i] {
		case PLAYER_X:
			digit = 1
		case PLAYER_O:
			digit = 2
		}
		key += digit * factor
		factor *= 3
	}

	return key
}

rotate_board_clockwise :: proc(board: [BOARD_SIZE]int) -> (rotated: [BOARD_SIZE]int) {
	rotated[0] = board[6]
	rotated[1] = board[3]
	rotated[2] = board[0]
	rotated[3] = board[7]
	rotated[4] = board[4]
	rotated[5] = board[1]
	rotated[6] = board[8]
	rotated[7] = board[5]
	rotated[8] = board[2]
	return
}

reflect_board :: proc(board: [BOARD_SIZE]int) -> (reflected: [BOARD_SIZE]int) {
	reflected[0] = board[2]
	reflected[1] = board[1]
	reflected[2] = board[0]
	reflected[3] = board[5]
	reflected[4] = board[4]
	reflected[5] = board[3]
	reflected[6] = board[8]
	reflected[7] = board[7]
	reflected[8] = board[6]
	return
}

canonical_board_key :: proc(board: [BOARD_SIZE]int) -> int {
	best_key := board_key(board)
	rotated := board

	for rotation := 0; rotation < 4; rotation += 1 {
		rotated_key := board_key(rotated)
		if rotated_key < best_key {
			best_key = rotated_key
		}

		reflected := reflect_board(rotated)
		reflected_key := board_key(reflected)
		if reflected_key < best_key {
			best_key = reflected_key
		}

		rotated = rotate_board_clockwise(rotated)
	}

	return best_key
}

minimax_cache_index :: proc(board: [BOARD_SIZE]int, player: int) -> int {
	offset := 0
	if player == PLAYER_O {
		offset = 1
	}
	return board_key(board) * 2 + offset
}

negamax_score :: proc(board: [BOARD_SIZE]int, player: int, cache: ^Minimax_Cache) -> int {
	cache_index := minimax_cache_index(board, player)
	if cache.known[cache_index] {
		return cache.scores[cache_index]
	}

	winner := check_winner(board)
	if winner != EMPTY {
		score := -1
		if winner == player {
			score = 1
		}
		cache.known[cache_index] = true
		cache.scores[cache_index] = score
		return score
	}

	if board_is_full(board) {
		cache.known[cache_index] = true
		cache.scores[cache_index] = 0
		return 0
	}

	best_score := -2
	for move := 0; move < BOARD_SIZE; move += 1 {
		if !is_legal_move(board, move) {
			continue
		}

		next_board := board
		next_board[move] = player
		score := -negamax_score(next_board, other_player(player), cache)
		if score > best_score {
			best_score = score
		}
	}

	cache.known[cache_index] = true
	cache.scores[cache_index] = best_score
	return best_score
}

best_move_for_player :: proc(
	board: [BOARD_SIZE]int,
	player: int,
	cache: ^Minimax_Cache,
) -> (
	best_move: int,
	best_score: int,
) {
	best_move = -1
	best_score = -2

	for move := 0; move < BOARD_SIZE; move += 1 {
		if !is_legal_move(board, move) {
			continue
		}

		next_board := board
		next_board[move] = player
		score := -negamax_score(next_board, other_player(player), cache)
		if score > best_score || (score == best_score && (best_move == -1 || move < best_move)) {
			best_move = move
			best_score = score
		}
	}

	return
}

best_move_label :: proc(
	board: [BOARD_SIZE]int,
	player: int,
	cache: ^Minimax_Cache,
) -> (
	label: [BOARD_SIZE]f32,
	best_move: int,
	best_score: int,
) {
	best_move, best_score = best_move_for_player(board, player, cache)
	if best_move >= 0 {
		label[best_move] = 1
	}
	return
}

collect_dataset :: proc(
	board: [BOARD_SIZE]int,
	player: int,
	visited: ^[UNIQUE_BOARD_STATES]bool,
	cache: ^Minimax_Cache,
	dataset: ^Dataset,
) {
	if is_terminal_board(board) {
		return
	}

	state_key := canonical_board_key(board)
	if visited[state_key] {
		return
	}
	visited[state_key] = true

	input := board_to_input(board, player)
	label, _, _ := best_move_label(board, player, cache)
	append(&dataset.inputs, input)
	append(&dataset.labels, label)

	next_player := other_player(player)
	for move := 0; move < BOARD_SIZE; move += 1 {
		if !is_legal_move(board, move) {
			continue
		}

		next_board := board
		next_board[move] = player
		collect_dataset(next_board, next_player, visited, cache, dataset)
	}
}

generate_dataset :: proc() -> Dataset {
	dataset := Dataset{}
	visited := [UNIQUE_BOARD_STATES]bool{}
	cache := new(Minimax_Cache)
	defer free(cache)
	collect_dataset(make_empty_board(), PLAYER_X, &visited, cache, &dataset)
	return dataset
}

choose_filtered_move :: proc(
	board: [BOARD_SIZE]int,
	output: []f32,
) -> (
	move: int,
	confidence: f32,
) {
	move = -1
	confidence = 0

	total := f32(0)
	best_value := f32(-1)
	fallback_move := -1

	for i := 0; i < BOARD_SIZE; i += 1 {
		if !is_legal_move(board, i) {
			continue
		}

		if fallback_move == -1 {
			fallback_move = i
		}

		value := output[i]
		if value < 0 {
			value = 0
		}
		total += value
		if value > best_value {
			best_value = value
			move = i
		}
	}

	if move == -1 {
		return fallback_move, 0
	}

	if total > 0 {
		confidence = best_value / total
	} else {
		move = fallback_move
		confidence = 0
	}

	return
}
