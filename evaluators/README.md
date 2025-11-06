### Missing Sudoku Evaluator

This directory is missing the `sudoku.py` evaluator script.

The official `pretrain.py` script fails to evaluate the `trm_sudoku.pt` model because it cannot find an evaluator named "sudoku". The log shows a `No evaluator found` warning, and the script exits without printing an accuracy score.

This makes it impossible to reproduce the paper's Sudoku evaluation results with the provided code.

