# AQFChallengeCode
(Not terrible) Air Quality Forecasting Challenge code

## How to run
### Data
- Put data folder in the folder named "data" such that:
    + Training data path is "data/data-train"
    + Testing data path is "data/public-test"
- Instead, the data can be put anywhere and the path will be specify by the flag --train-path and --test-path in the corresponding files

### Using the model
- Run 2 files forecaster.py and extrapolator.py to train the model using training data
- After that, run e2e_model.py to generate the output to submit.

## Available flags (optional):
### forecaster.py, extrapolator.py
- --train-path: Path of the training data folder
- --test-rate: Ratio of the test dataset for evaluating
- --learning-rate: Learning rate of the forecaster
- --epochs: Number of epochs to train (1-3 epochs are recommended)
- --batch-size: Batch size for a single training step
- For a complete list of the flags and its default value, run `python forecaster.py --help`, `python extrapolator.py --help`

### e2e_model.py
- --test-path: Path of the testing data folder
- --output-path: Path to the temporary output folder
- For a complete list of the flags and its default value, run `python e2e_model.py --help`