# sunwoo-cau
daq_test.py
This script is used to acquire voltage signals from PVDF strain sensors via a National Instruments DAQ device. It sets up the acquisition session, samples multiple channels simultaneously, and saves the collected raw data for further processing.

signalcollectregression.py
This script extracts meaningful impact segments from the raw sensor data obtained by daq_test.py. Each impact event is segmented into a 300-sample window (50 pre-trigger + 250 post-trigger), and annotated with the actual impact coordinates (x, y). The result is a structured dataset suitable for regression training.

gptcnnregression.py
This is the main training pipeline. It converts the time-series data into GPT (Grid Pattern Transformation) images, then feeds them into a CNN regression model. The model learns to predict the impact coordinates (x, y) directly. Training performance is evaluated using MAE and MSE metrics.

carbon619_regression2.csv
This file is the output of signalcollectregression.py. It contains multiple labeled impact samples, each with PVDF sensor responses and corresponding (x, y) coordinates. It is used as the input dataset for the CNN regression model.



Wang, M., Yan, Y., Zhang, W., Zhang, Y., Wu, D., Wang, Y., Qing, X., & Wang, Y. (2025).
An impact localization method for composite structures based on time series features and machine learning.
Composite Structures.




result:
![regression_training_curve](https://github.com/user-attachments/assets/2fdbba5f-3da2-4139-b40a-6694d23b88c1)

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 48, 98, 32)          │           1,184 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 24, 49, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 22, 47, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 11, 23, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 16192)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 16192)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 32)                  │         518,176 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 2)                   │              66 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 537,922 (2.05 MB)
 Trainable params: 537,922 (2.05 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 6s 29ms/step - loss: 251.4056 - mae: 9.3728 - val_loss: 98.5215 - val_mae: 4.7229
Epoch 2/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 28ms/step - loss: 78.1461 - mae: 3.6129 - val_loss: 66.4197 - val_mae: 2.7598
Epoch 3/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 29ms/step - loss: 51.2460 - mae: 2.4909 - val_loss: 61.8392 - val_mae: 3.1191
Epoch 4/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 6s 33ms/step - loss: 45.2507 - mae: 2.2970 - val_loss: 46.7878 - val_mae: 2.1695
Epoch 5/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 29ms/step - loss: 33.5955 - mae: 1.8176 - val_loss: 42.6624 - val_mae: 2.1879
Epoch 6/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 28.4023 - mae: 1.5700 - val_loss: 44.8844 - val_mae: 2.1147
Epoch 7/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 24.8638 - mae: 1.4664 - val_loss: 36.9022 - val_mae: 1.8983
Epoch 8/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 22.9213 - mae: 1.2906 - val_loss: 48.1955 - val_mae: 2.1497
Epoch 9/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 22.6004 - mae: 1.2938 - val_loss: 37.0738 - val_mae: 1.8323
Epoch 10/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 22.8903 - mae: 1.3550 - val_loss: 34.7394 - val_mae: 1.7355
Epoch 11/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 18.6447 - mae: 1.0621 - val_loss: 33.3261 - val_mae: 1.7042
Epoch 12/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 16.6709 - mae: 1.0505 - val_loss: 35.2778 - val_mae: 1.6815
Epoch 13/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 17.2219 - mae: 1.0126 - val_loss: 39.1655 - val_mae: 2.0563
Epoch 14/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 14.5462 - mae: 0.9360 - val_loss: 32.9632 - val_mae: 1.5685
Epoch 15/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 14.7810 - mae: 0.9586 - val_loss: 42.9379 - val_mae: 1.9796
Epoch 16/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 15.5385 - mae: 1.0306 - val_loss: 41.7065 - val_mae: 1.8461
Epoch 17/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 19.5446 - mae: 1.1734 - val_loss: 36.6876 - val_mae: 1.6758
Epoch 18/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 26ms/step - loss: 13.0078 - mae: 0.8506 - val_loss: 34.9607 - val_mae: 1.6595
Epoch 20/20
171/171 ━━━━━━━━━━━━━━━━━━━━ 5s 26ms/step - loss: 15.0376 - mae: 1.0144 - val_loss: 32.2491 - val_mae: 1.5976
37/37 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 29.6597 - mae: 1.5452
✅ 평균 위치 오차 (MAE): 1.60 mm
