import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Add, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generateData(trainDataFolder, drop=[], testRatio=0.1):
  loc_input = pd.read_csv(f"{trainDataFolder}/location_input.csv", index_col=0)
  loc_output = pd.read_csv(f"{trainDataFolder}/location_output.csv", index_col=0)
  stations = []
  locations = []
  coords_input = []
  coords_output = []
  for i in range(len(loc_input)):
    stations.append(pd.read_csv(f"{trainDataFolder}/input/{loc_input['station'][i]}.csv", index_col=0))
    coords_input.append((loc_input["longitude"][i], loc_input["latitude"][i]))
  
  for i in range(len(loc_output)):
    if i in drop:
      continue
    locations.append(pd.read_csv(f"{trainDataFolder}/output/{loc_output['station'][i]}.csv", index_col=0))
    coords_output.append((loc_output["longitude"][i], loc_output["latitude"][i]))

  X = []
  y = []
  
  for i in range((9000)):
    data_piece = []
    for station in stations:
      data_piece.append(station["PM2.5"][i])
    data_piece.extend((0, 0))
    coords = []
    labels = []
    for j, dp in enumerate(data_piece[:-2]):
      if np.isnan(dp):
        continue
      coords.append(coords_input[j])
      labels.append(dp)
    
    for j in range(len(coords_output)):
      t = locations[j]["PM2.5"][i]
      if np.isnan(t):
        continue
      coords.append(coords_output[j])
      labels.append(t)
    for coord, label in zip(coords, labels):
      data_piece[-2:] = coord
      X.append(data_piece.copy())
      y.append(label)
      
  X = np.nan_to_num(X, nan=-1)
  y = np.nan_to_num(y, nan=-1)
  X[:,-2] = 10*(X[:,-2] - 105.8)
  X[:,-1] = 10*(X[:,-1] - 21)
  
  X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=testRatio)
  return X_tr, y_tr, X_t, y_t


def dataAugment(X_tr, Y_tr, mask_rate, rate, strength, pools=1):
  t = strength*np.sqrt(12)/2
  n = X_tr.shape[0]
  for i in range(pools):
    idx = np.random.choice(n, int(n*rate), replace=False)
    augX = X_tr[idx].copy()
    augY = Y_tr[idx].copy()
    mask = np.random.random(augX.shape) < mask_rate
    mask[:,-2:] = False
    mask2 = np.random.random(augX.shape) > rate
    delta = np.random.uniform(-t, t, augX.shape)
    delta[:,-2:] = np.random.uniform(-t/100, t/100, (augX.shape[0], 2))
    delta[mask2] = 0
    augX += delta
    augX[mask] = -1
    mask2[mask] = 0
    X_tr = np.concatenate((X_tr, augX))
    Y_tr = np.concatenate((Y_tr, augY))
  return X_tr, Y_tr


def getModel(lr=0.001):
  input_layer = Input(shape=(13,))
  x = Dense(128, activation='relu')(input_layer)
  x1 = Dense(256, activation='relu')(x)
  x2 = Concatenate()([x, x1])
  x2 = Dense(256, activation='relu')(x2)
  x3 = Dense(256, activation='relu')(x2)
  x4 = Dense(256, activation='relu')(x3)
  x4 = Dense(256, activation='relu')(x4)
  x4 = Dense(256, activation='relu')(x4)
  x5 = Add()([x3, x4])
  x5 = Dense(256, activation='relu')(x5)
  x5 = Dense(256, activation='relu')(x5)
  x6 = Add()([x2, x5])
  x6 = Dense(256, activation='relu')(x6)
  x6 = Dense(128, activation='relu')(x6)
  y = Dense(1)(x6)
  model = Model(inputs=[input_layer], outputs=y)
  model.compile(optimizer=Adam(learning_rate=lr), loss='huber', metrics=['mae', 'mape'])
  return model


def evaluate(model, X_test, y_test):
  if len(X_test) == 0:
    print("No data to evaluate")
    return
  lossF = Huber()
  y_pred = model.predict(X_test).flatten()
  loss = lossF(y_test, y_pred)
  print("loss: ", loss.numpy())
  ae = np.abs(y_pred - y_test)
  print("mae:", np.mean(ae))
  ape = ae/(y_test + 1e-15)
  print("mape:", np.mean(ape)*100)
  print("mdape:", np.median(ape)*100)


def main(args):
  drop = []
  p = 1
  for i in range(11):
    if (args.drop_mask & p) != 0:
      drop.append(i)
    p *= 2
  print("Initializing...")
  X_train, y_train, X_test, y_test = generateData(args.train_path, drop=drop, testRatio=args.test_rate)
  X_tr, y_tr = dataAugment(X_train, y_train, mask_rate=args.augment_mask_rate, rate=args.augment_rate, strength=args.augment_strength, pools=args.augment_pools)

  print("Training...")
  model = getModel(lr=args.learning_rate)
  model.fit(X_tr, y_tr, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.validation_split)

  print("Evaluating...")
  evaluate(model, X_test, y_test)

  if not os.path.exists("weights"):
    os.mkdir("weights")

  model.save_weights("weights/extrapolator.h5")
  print("Weights saved in weights/extrapolator.h5")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-path", type=str, default="./data/data-train", help="Path of the training data folder (default: in the sample folder of this fil)")
  parser.add_argument("--test-rate", type=float, default=0.1, help="Ratio of the test dataset for evaluating (default: 0.1)")
  parser.add_argument("--augment-rate", type=float, default=0.6, help="Rate at which the data are augmented by noise for one run (default: 0.6)")
  parser.add_argument("--augment-mask-rate", type=float, default=0.01, help="Rate at which the station data are masked (default: 0.01)")
  parser.add_argument("--augment-strength", type=float, default=2, help="Standard deviation of the uniform noise added to the augmented data (default: 2)")
  parser.add_argument("--augment-pools", type=int, default=5, help="Number of times apply augmentation to the data (default: 5)")
  parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the forecaster (default: 0.001)")
  parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train(default: 15)")
  parser.add_argument("--verbose", type=bool, default=True, help="Whether viewing the training process or not (default: True)")
  parser.add_argument("--batch-size", type=int, default=32, help="Batch size for a single training step (default: 32)")
  parser.add_argument("--validation-split", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
  parser.add_argument("--drop-mask", type=int, default=0, help="(Advanced) Bitmask indicating which stations will be excluded from data. (default: 0 - nomask)")
  args = parser.parse_args()
  main(args)
  