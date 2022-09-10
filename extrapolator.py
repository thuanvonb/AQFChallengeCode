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
from tqdm import tqdm
from tensorflow.keras.utils import Sequence

class Generator(Sequence):
  def __init__(self, datapieces, coords, length, batch_size):
    self.data_piece = datapieces
    self.coords_input = coords
    self.length = length
    self.batch_size = batch_size


  def on_epoch_end(self):
    pass


  def __len__(self):
    return self.length


  def __getitem__(self, x):
    batch_idxs = np.random.choice(len(self.data_piece), self.batch_size)
    X = []
    y = []
    for batchId in batch_idxs:
      data_piece = self.data_piece[batchId]
      coords = self.coords_input
      while True:
        idxs = np.random.choice(71, 9, replace=False)
        if np.isnan(data_piece[idxs[-1]]):
          batchId = np.random.choice(len(self.data_piece))
          data_piece = self.data_piece[batchId]
          continue
        break
      x = []
      p = data_piece[idxs[-1]]
      for k, idx in enumerate(idxs):
        if k < 8:
          x.append(data_piece[idx])
        x.extend(coords[idx])
      X.append(x)
      y.append(p)
    X = np.nan_to_num(X, nan=-1)
    y = np.array(y)
    return X, y



def generateData(trainDataFolder, drop=[]):
  loc_input = pd.read_csv(f"{trainDataFolder}/air/location.csv", index_col=0)
  # loc_output = pd.read_csv(f"{trainDataFolder}/location_output.csv", index_col=0)
  stations = []
  coords_input = []
  for i in range(len(loc_input)):
    stations.append(pd.read_csv(f"{trainDataFolder}/air/{loc_input['location'][i]}.csv", index_col=0))
    coords_input.append((loc_input["longitude"][i]-105.8, loc_input["latitude"][i]-21))

  X = []
  y = []
  
  datapieces = []
  for i in tqdm(range((6000))):
    data_piece = []
    for j, station in enumerate(stations):
      data_piece.append(station["PM2.5"][i])
    datapieces.append(data_piece)

  return datapieces, coords_input


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
    X_tr = np.concatenate((X_tr, augX))
    Y_tr = np.concatenate((Y_tr, augY))
  return X_tr, Y_tr


def getModel(lr=0.001):
  input_layer = Input(shape=(26,))
  x = Dense(1024, activation='relu')(input_layer)
  x1 = Dense(1024, activation='relu')(x)
  x2 = Concatenate()([x, x1])
  x2 = Dense(1024, activation='relu')(x2)
  x3 = Dense(1024, activation='relu')(x2)
  x3 = Dense(1024, activation='relu')(x3)
  x4 = Dense(1024, activation='relu')(x3)
  x4 = Dense(1024, activation='relu')(x4)
  x4 = Dense(1024, activation='relu')(x4)
  x4 = Dense(1024, activation='relu')(x4)
  x4 = Dense(1024, activation='relu')(x4)
  x5 = Add()([x3, x4])
  x5 = Dense(1024, activation='relu')(x5)
  x5 = Dense(1024, activation='relu')(x5)
  x6 = Add()([x2, x5])
  x6 = Dense(512, activation='relu')(x6)
  x6 = Dense(512, activation='relu')(x6)
  y = Dense(1)(x6)
  model = Model(inputs=[input_layer], outputs=y)
  model.compile(optimizer=Adam(learning_rate=lr), loss='huber', metrics=['mae', 'mape'])
  model.summary()
  return model


def evaluate(model, generator):
  lossF = Huber()
  y_pred = model.predict(X_test).flatten()
  loss = lossF(y_test, y_pred)
  print("loss: ", loss.numpy())
  ae = np.abs(y_pred - y_test)
  print("mae:", np.mean(ae))
  ape = ae/(y_test + 1e-15)
  print("mape:", np.mean(ape)*100)
  print("mdape:", np.median(ape)*100)


def normalize(X, mean=None, std=None):
  X_out = X.copy()

  if mean is None and std is None:
    mean = np.nanmean(X_out, axis=0)
    std = np.nanstd(X_out, axis=0)
    X_out = (X_out - mean) / std
    return X_out, mean, std

  X_out = (X_out - mean) / std
  return X_out


def main(args):
  print("Initializing...")
  datapieces, coords = generateData(args.train_path, drop=[])
  trainGen = Generator(datapieces, coords, 100000, args.batch_size)
  valGen = Generator(datapieces, coords, 100, args.batch_size)
  testGen = Generator(datapieces, coords, 100, args.batch_size)

  print("Training...")
  model = getModel(lr=args.learning_rate)
  model.fit(trainGen, epochs=args.epochs, validation_data=valGen, validation_steps=len(valGen))

  # print("Evaluating...")
  # evaluate(model, testGen)

  if not os.path.exists("weights"):
    os.mkdir("weights")

  model.save_weights("weights/extrapolator.h5")
  print("Weights saved in weights/extrapolator.h5")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-path", type=str, default="./data/data-train", help="Path of the training data folder (default: in the sample folder of this fil)")
  parser.add_argument("--test-rate", type=float, default=0.1, help="Ratio of the test dataset for evaluating (default: 0.1)")
  parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the forecaster (default: 0.001)")
  parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train(default: 3)")
  parser.add_argument("--verbose", type=bool, default=True, help="Whether viewing the training process or not (default: True)")
  parser.add_argument("--batch-size", type=int, default=128, help="Batch size for a single training step (default: 128)")
  parser.add_argument("--validation-split", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
  # parser.add_argument("--drop-mask", type=int, default=0, help="(Advanced) Bitmask indicating which stations will be excluded from data. (default: 0 - nomask)")
  args = parser.parse_args()
  main(args)
  