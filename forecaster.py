import pandas as pd
import numpy as np

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam


def verifyNaN(d):
  return (~np.isnan(d)).all()


class WeightedLoss(tf.keras.losses.Loss):
  def __init__(self, alpha, sampleWeight, windowLength, offsetLength, **kwargs):
    super().__init__(**kwargs)
    assert 0 < sampleWeight < 1/(windowLength + offsetLength), "invalid sample weight (0, %.6f)" % 1/(windowLength + offsetLength)
    self.sampleWeight = sampleWeight
    self.windowLength = windowLength
    self.offsetLength = offsetLength
    self.growthRate = self._solveForF(sampleWeight, err=1e-9)
    self.alpha = alpha
    self.weights = np.zeros(windowLength + offsetLength, dtype=np.float32)
    self.weights[-offsetLength:] = np.linspace(1, offsetLength, offsetLength)
    self.weights = self.sampleWeight * np.power(self.growthRate, self.weights)


  def _solveForF(self, t, err=1e-6):
    c = 1/t - (self.windowLength-1)
    f = lambda x: (1 - np.power(x, self.offsetLength+1))/(1 - x) - c
    df = lambda x: (self.offsetLength*np.power(x, self.offsetLength+1) - \
                    (self.offsetLength+1)*np.power(x, self.offsetLength) + 1) / np.power(x - 1, 2)
    sample = 1.3
    while True:
      v = f(sample)
      if (abs(v) < err):
        return sample
      dv = df(sample)
      sample -= v/dv


  def call(self, y_true, y_pred):
    err = y_true - y_pred
    errA = tf.math.abs(err)
    err2 = err**2
    err = tf.where(errA <= self.alpha, err2/2, self.alpha*(errA - self.alpha/2))
    return tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(err, axis=-1)*self.weights, axis=-1))


def getModel2(lr=0.001, wl_sample_weight=0.008):
  x = Input(shape=(None, 11))
  x1 = Dense(256, activation='relu')(x)
  x2 = Concatenate()([x, x1])
  x2 = Dense(256, activation='relu')(x2)
  x3 = Dense(256, activation='relu')(x2)
  x3 = GRU(200, return_sequences=True)(x3)
  x3 = Dense(256, activation='relu')(x3)
  x4 = Add()([x2, x3])
  x4 = Dense(256, activation='relu')(x4)
  x5 = Dense(256, activation='relu')(x4)
  x5 = GRU(200, return_sequences=True)(x5)
  x5 = Dense(256, activation='relu')(x5)
  x6 = Add()([x4, x5])
  x6 = Dense(256, activation='relu')(x6)
  x7 = Dense(256, activation='relu')(x6)
  y = Dense(3)(x7)
  model = Model(inputs=[x], outputs=y)
  model.compile(optimizer=Adam(learning_rate=lr), loss=WeightedLoss(2, wl_sample_weight, 24, 23),\
                metrics=['mae', 'mape'])
  return model


def createData(name, meteoData, ts=24, fc=24):
  csv = pd.read_csv(name, usecols=['time', 'PM2.5', 'humidity', 'temperature'])
  csv["time"] = pd.to_datetime(csv["time"], format="%Y-%m-%d %H:%M:%S").apply(lambda x: x.hour)
  csv["time_sin"] = csv["time"].apply(lambda x: np.sin(x*np.pi/12))
  csv["time_cos"] = csv["time"].apply(lambda x: -np.cos(x*np.pi/12))
  csv["n1"] = np.ones(len(csv))*-1
  data = csv.drop(["time"], axis=1).to_numpy()
  n = len(data)
  X = []
  Y = []
  for i in range(n):
    if i + ts + fc > n:
      break
    x1 = data[i:i+ts, :6]
    x2 = meteoData[i:i+ts]
    x = np.concatenate([x2, x1], axis=1)
    zeros = np.zeros((fc-1, 11))
    zeros[:,8] = data[i+ts:i+ts+fc-1,3]
    zeros[:,9] = data[i+ts:i+ts+fc-1,4]
    zeros[:,10] = 1
    x = np.concatenate((x, zeros), axis=0)
    y = data[i+1:i+ts+fc, :3]
    if verifyNaN(x) and verifyNaN(y):
      X.append(x)
      Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y


def bigDataset(trainPath, timestampSize=24, forecastCapacity=24, test_rate=0.1): 
  meteoData = pd.read_csv(f"{trainPath}/meteo/station_86.csv", index_col=0).drop(["time"], axis=1).to_numpy()
  x = []
  for i in range(3):
    k = meteoData[:-1]*(3-i)/3 + meteoData[1:]*i/3
    x.append(np.expand_dims(k, axis=1))
  meteoData = np.concatenate(x, axis=1).reshape((-1, 5))

  training = os.listdir(f"{trainPath}/air/")
  del training[training.index("location.csv")]
  # training.extend(os.listdir(f"{trainPath}/output/"))
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []
  for i, trainingFile in enumerate(training):
    filePath = f"{trainPath}/air/"
    X, Y = createData(filePath + trainingFile, meteoData, timestampSize, forecastCapacity)
    n = len(X)
    tId = int(n * (1-test_rate))
    X_test.append(X[tId:])
    Y_test.append(Y[tId:])
    X_train.append(X[:tId])
    Y_train.append(Y[:tId])
  X_train = np.concatenate(X_train, axis=0)
  Y_train = np.concatenate(Y_train, axis=0)
  X_test = np.concatenate(X_test, axis=0)
  Y_test = np.concatenate(Y_test, axis=0)
  return X_train, Y_train, X_test, Y_test


def dataAugment(X, Y, rate, strength, pools=1):
  t = strength*np.sqrt(12)/2
  n = X.shape[0]
  for i in range(pools):
    idx = np.random.choice(n, int(n*rate), replace=False)
    aug = X[idx].copy()
    augY = Y[idx].copy()
    mask = np.random.random(aug.shape) < rate
    delta = np.random.uniform(-t, t, aug.shape)
    delta[mask] = 0
    delta[:,:,8:] = 0
    delta[:,24:,:] = 0

    aug += delta
    X = np.concatenate((X, aug))
    Y = np.concatenate((Y, augY))
  return X, Y


def normalize(X, mean=None, std=None):
  X_out = X.copy()

  if mean is None and std is None:
    mean = np.mean(np.mean(X_out[:,:24,:-3], axis=1), axis=0)
    std = np.std(np.std(X_out[:,:24,:-3], axis=1), axis=0)
    X_out[:,:24,:-3] = (X_out[:,:24,:-3] - mean) / std
    return X_out, mean, std

  X_out[:,:24,:-3] = (X_out[:,:24,:-3] - mean) / std
  return X_out


def train(model, X_tr, y_tr, epochs=1, verbose=1, batch_size=16, validation_split=0.0):
  print("Training...")
  model.fit(X_tr, y_tr, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=args.validation_split)


def evaluate(model, X_t, y_t):
  if len(X_t) == 0:
    print("No data to evaluate")
    return
  lossF = WeightedLoss(2, 0.008, 24, 23)
  print("Evaluating...")
  y_p = model.predict(X_t)
  loss = lossF(y_t, y_p)
  print("loss: ", loss.numpy())
  ae = np.abs(y_t - y_p)
  ae_f = np.abs(y_t[:,24:,:] - y_p[:,24:,:])
  print("mae (all):", np.mean(ae))
  print("mae (forecasting):", np.mean(ae_f))
  ape = ae/(y_t + 1e-15)
  ape_f = ae_f/(y_t[:,24:,:] + 1e-15)
  print("mape (all):", np.mean(ape)*100)
  print("mape (forecasting):", np.mean(ape_f)*100)
  print("mdape (all):", np.median(ape)*100)
  print("mdape (forecasting):", np.median(ape_f)*100)


def main(args):
  print("Initializing...")
  X_tr, y_tr, X_t, y_t = bigDataset(args.train_path, test_rate=args.test_rate)
  print("Data collected:", (X_tr.shape[0] + X_t.shape[0]))
  print("Data used for training (before augmented):", X_tr.shape[0])
  X_tr, mean_tr, std_tr = normalize(X_tr)
  X_t = normalize(X_t, mean_tr, std_tr)
  X_tr, y_tr = dataAugment(X_tr, y_tr, rate=args.augment_rate, strength=args.augment_strength, pools=args.augment_pools)
  print("Data used for training (after augmented):", X_tr.shape[0])

  model = getModel2(args.learning_rate, args.wl_sample_weight)
  train(model, X_tr, y_tr, epochs=args.epochs, verbose=args.verbose, batch_size=args.batch_size, validation_split=args.validation_split)
  evaluate(model, X_t, y_t)

  if not os.path.exists("weights"):
    os.mkdir("weights")
  if not os.path.exists("paras"):
    os.mkdir("paras")

  model.save_weights("weights/forecaster.h5")
  print("Weights saved in weights/forecaster.h5")
  np.savez("paras/forecaster", mean=mean_tr, std=std_tr)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-path", type=str, default="./data/data-train", help="Path of the training data folder (default: in the sample folder of this fil)")
  parser.add_argument("--test-rate", type=float, default=0.1, help="Ratio of the test dataset for evaluating (default: 0.1)")
  parser.add_argument("--augment-rate", type=float, default=0.3, help="Rate at which the data are augmented by noise for one run (default: 0.3)")
  parser.add_argument("--augment-strength", type=float, default=0.1, help="Standard deviation of the uniform noise added to the augmented data (default: 0.1)")
  parser.add_argument("--augment-pools", type=int, default=7, help="Number of times apply augmentation to the data (default: 7)")
  parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the forecaster (default: 0.001)")
  parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train (1-3 epochs are recommended) (default: 1)")
  parser.add_argument("--verbose", type=bool, default=True, help="Whether viewing the training process or not (default: True)")
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size for a single training step (default: 16)")
  parser.add_argument("--validation-split", type=float, default=0.0, help="Validation set ratio, only meaningful when training multiple epochs")
  parser.add_argument("--wl-sample-weight", type=float, default=0.008, help="(Advanced) Weight of the loss of the sample data part for RNN contributing to the whole training loss (must be between 0 and 0.02, exclusive) (default: 0.008)")
  args = parser.parse_args()
  main(args)
  # model = getModel2(args.learning_rate, args.wl_sample_weight)
  # model.summary()
  