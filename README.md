# SCAI_Hedge_your_bets

## Sports Betting ML Project

This project focuses on developing a machine learning model for sports betting predictions, specifically for moneyline bets which bets on a winner between two teams.

### Files

#### 1. `train_test.py`

This script initiates the training and testing process for the machine learning model. It includes functions for training the model using historical data, evaluating its performance, and saving the trained model to a `.pth` file.

#### 2. `process.py`

This file contains a library of functions for processing and preparing data. It is utilized in various parts of the project for data preprocessing tasks.

#### 3. `model.py`

The model.py file contains the definition of the machine learning model class along with associated functions. 

#### 4. `fetch_games.py`

This script fetches and formats live game data from an API. It is designed to be run via the command line.

#### 5. `predict.py`

The predict.py script makes predictions on formatted data obtained from fetch_games.py. It is designed to be run via the command line.

#### 6. `run_script.py`

The run_script.py script serves as a higher-level script that runs fetch_games.py and predict.py as subprocesses. This is useful for automating the entire data-fetching and prediction process.

### Libraries

The project utilizes the following libraries:

- `os`
- `pandas`
- `torch`
- `torch.nn`
- `torch.utils.data.Dataset`
- `sklearn.preprocessing.MinMaxScaler`
- `torch.utils.data.DataLoader`
- `matplotlib.pyplot`
- `seaborn`
- `sklearn.metrics.confusion_matrix`
- `torch.optim`

### Data

Had issues with data scraping. So we used pre-scraped box scores from https://basketball-reference.com/.

You can download same dataset here https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblVtX3hYQ1VlZGpvOWFra2pCa2stYUE2bUpXd3xBQ3Jtc0tua0VyUVRmRkFyM3cwQ3F2V0xQaWhEX21wTEJNQ0dhZ0ZfSTAyZUJzeFUtcTM3YU8xV0JrVk9jNzRHTC02QjEtc20zWW4tM1BTNVhpc2hOem5rQ3E4elg5eUo4MEVWWElFR202dU94RUlYd290VFhWcw&q=https%3A%2F%2Fdrive.google.com%2Fuc%3Fexport%3Ddownload%26id%3D1YyNpERG0jqPlpxZvvELaNcMHTiKVpfWe&v=egTylm6C2is.

Live game data was fetched using this api https://github.com/swar/nba_api .
