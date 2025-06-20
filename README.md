# sunwoo-cau
daq_test.py
This script is used to acquire voltage signals from PVDF strain sensors via a National Instruments DAQ device. It sets up the acquisition session, samples multiple channels simultaneously, and saves the collected raw data for further processing.

signalcollectregression.py
This script extracts meaningful impact segments from the raw sensor data obtained by daq_test.py. Each impact event is segmented into a 300-sample window (50 pre-trigger + 250 post-trigger), and annotated with the actual impact coordinates (x, y). The result is a structured dataset suitable for regression training.

gptcnnregression.py
This is the main training pipeline. It converts the time-series data into GPT (Grid Pattern Transformation) images, then feeds them into a CNN regression model. The model learns to predict the impact coordinates (x, y) directly. Training performance is evaluated using MAE and MSE metrics.

carbon619_regression.csv
This file is the output of signalcollectregression.py. It contains multiple labeled impact samples, each with PVDF sensor responses and corresponding (x, y) coordinates. It is used as the input dataset for the CNN regression model.

Wang, M., Yan, Y., Zhang, W., Zhang, Y., Wu, D., Wang, Y., Qing, X., & Wang, Y. (2025).
An impact localization method for composite structures based on time series features and machine learning.
Composite Structures.
