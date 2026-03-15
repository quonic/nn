package tictactoe_example

import nn "../../"
import "core:bufio"
import "core:fmt"
import "core:io"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:strings"

print_training_progress :: proc(progress: nn.Training_Progress) {
	metrics := progress.metrics
	fmt.printf(
		"epoch=%d/%d loss=%0.5f accuracy=%0.2f%%\n",
		progress.epoch,
		progress.total_epochs,
		metrics.average_loss,
		metrics.accuracy * 100,
	)
}

brain_matches_example_setup :: proc(
	brain: ^nn.Brain,
	architecture: []int,
	config: nn.Brain_Config,
) -> bool {
	if len(brain.layers) != len(architecture) {
		return false
	}

	if brain.config.hidden_activation != config.hidden_activation ||
	   brain.config.output_activation != config.output_activation ||
	   brain.config.loss != config.loss {
		return false
	}

	for i := 0; i < len(architecture); i += 1 {
		if len(brain.layers[i].neurons) != architecture[i] {
			return false
		}
	}

	return true
}

print_report :: proc(brain: ^nn.Brain, inputs, labels: [][]f32) {
	report, ok := nn.compute_classification_report(brain, inputs, labels)
	if !ok {
		return
	}
	defer nn.destroy_classification_report(&report)

	fmt.println("confusion matrix (rows=expected, cols=predicted):")
	for row := 0; row < report.class_count; row += 1 {
		for col := 0; col < report.class_count; col += 1 {
			if col > 0 {
				fmt.print(" ")
			}
			fmt.printf("%d", report.confusion_matrix[row * report.class_count + col])
		}
		fmt.println()
	}
}

read_line :: proc(scanner: ^bufio.Scanner) -> (string, bool) {
	if !bufio.scan(scanner) {
		return "", false
	}
	return strings.trim_space(bufio.scanner_text(scanner)), true
}

choose_human_side :: proc(scanner: ^bufio.Scanner) -> (side: int, ok: bool) {
	for {
		fmt.print("Choose your side [X/O]: ")
		line, line_ok := read_line(scanner)
		if !line_ok {
			return PLAYER_X, false
		}
		if len(line) == 0 {
			continue
		}

		switch line[0] {
		case 'x', 'X':
			return PLAYER_X, true
		case 'o', 'O':
			return PLAYER_O, true
		case:
			fmt.println("Enter X or O.")
		}
	}
}

read_human_move :: proc(scanner: ^bufio.Scanner, board: [BOARD_SIZE]int) -> (move: int, ok: bool) {
	for {
		fmt.print("Choose a move [1-9]: ")
		line, line_ok := read_line(scanner)
		if !line_ok {
			return -1, false
		}

		choice, parsed := strconv.parse_int(line)
		if !parsed || choice < 1 || choice > 9 {
			fmt.println("Enter a number from 1 to 9.")
			continue
		}

		move = choice - 1
		if !is_legal_move(board, move) {
			fmt.println("That square is already occupied.")
			continue
		}

		return move, true
	}
}

play_game :: proc(brain: ^nn.Brain) {
	stdin_reader, _ := io.to_reader(os.to_stream(os.stdin))
	scanner: bufio.Scanner
	bufio.scanner_init(&scanner, stdin_reader)
	defer bufio.scanner_destroy(&scanner)

	human_side, ok := choose_human_side(&scanner)
	if !ok {
		fmt.println("No terminal input available; skipping interactive game.")
		return
	}

	board := make_empty_board()
	player_to_move := PLAYER_X

	fmt.printf("You are %s. Squares use positions 1 through 9.\n", player_name(human_side))
	for {
		fmt.println()
		print_board(board)

		winner := check_winner(board)
		if winner != EMPTY {
			fmt.println()
			if winner == human_side {
				fmt.printf("You win as %s.\n", player_name(human_side))
			} else {
				fmt.printf("Model wins as %s.\n", player_name(winner))
			}
			return
		}

		if board_is_full(board) {
			fmt.println()
			fmt.println("Draw.")
			return
		}

		if player_to_move == human_side {
			move, move_ok := read_human_move(&scanner, board)
			if !move_ok {
				fmt.println("Input closed; ending game.")
				return
			}
			apply_move(&board, move, player_to_move)
		} else {
			input := board_to_input(board, player_to_move)
			output := nn.run(brain, input[:])
			move, confidence := choose_filtered_move(board, output)
			delete(output)

			if move < 0 {
				fmt.println("Model could not find a legal move.")
				return
			}

			apply_move(&board, move, player_to_move)
			fmt.printf(
				"Model plays %s at square %d (confidence=%0.2f%%).\n",
				player_name(player_to_move),
				move + 1,
				confidence * 100,
			)
		}

		player_to_move = other_player(player_to_move)
	}
}

main :: proc() {
	model_path := "tictactoe-model.json"
	architecture := [4]int{9, 32, 16, 9}
	config := nn.default_brain_config()
	config.hidden_activation = .ReLU
	config.output_activation = .Softmax
	config.loss = .Categorical_Cross_Entropy

	dataset := generate_dataset()
	defer destroy_dataset(&dataset)

	inputs := make([][]f32, len(dataset.inputs))
	labels := make([][]f32, len(dataset.labels))
	defer delete(inputs)
	defer delete(labels)

	for i := 0; i < len(dataset.inputs); i += 1 {
		inputs[i] = dataset.inputs[i][:]
		labels[i] = dataset.labels[i][:]
	}

	brain: nn.Brain
	loaded := false
	if os.exists(model_path) {
		brain, loaded = nn.load_brain(model_path)
		if loaded && !brain_matches_example_setup(&brain, architecture[:], config) {
			nn.destroy_brain(&brain)
			brain = {}
			loaded = false
			fmt.printf(
				"Ignoring %s because it does not match the current example configuration\n",
				model_path,
			)
		}
	}

	if !loaded {
		rand.reset_u64(42)
		brain = nn.make_brain_with_config(architecture[:], 0.10, config)
		fmt.printf("Generated %d perfect-play positions.\n", len(dataset.inputs))
		fmt.println("Training tic-tac-toe model...")

		training := nn.default_training_config()
		training.batch_size = 64
		training.shuffle_samples = true
		training.report_every = 100
		training.progress_callback = print_training_progress

		nn.train_with_config(&brain, inputs, labels, 400, training)

		if nn.save_brain(&brain, model_path) {
			fmt.printf("Saved trained model to %s\n", model_path)
		} else {
			fmt.printf("Warning: failed to save trained model to %s\n", model_path)
		}
	} else {
		fmt.printf("Loaded trained model from %s\n", model_path)
	}
	defer nn.destroy_brain(&brain)

	metrics := nn.compute_dataset_metrics(&brain, inputs, labels)
	fmt.printf(
		"final loss=%0.5f mse=%0.5f mae=%0.5f accuracy=%0.2f%%\n",
		metrics.average_loss,
		metrics.mean_squared_error,
		metrics.mean_absolute_error,
		metrics.accuracy * 100,
	)
	print_report(&brain, inputs, labels)

	fmt.println()
	fmt.println("Play against the model.")
	play_game(&brain)
}
