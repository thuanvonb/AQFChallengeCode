import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Input, Concatenate, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


stations = None
coords = None
para = None


def getModelForecaster(lr=0.001, wl_sample_weight=0.008):
  input_layer = Input(shape=(None, 6))
  x1 = Dense(128, activation='relu')(input_layer)
  x2 = GRU(200, return_sequences=True)(x1)
  x2 = Dense(256, activation='relu')(x2)
  x3 = Dense(256, activation='relu')(x2)
  x3 = Concatenate()([x2, x3])
  x3 = Dense(256, activation='relu')(x3)
  x3 = Dense(128, activation='relu')(x3)
  x4 = Add()([x1, x3])
  x4 = Dense(128, activation='relu')(x4)
  x4 = Dense(64, activation='relu')(x4)
  y = Dense(3)(x4)
  model = Model(inputs=[input_layer], outputs=y)
  model.compile(optimizer=Adam(learning_rate=lr), loss="huber", metrics=['mae', 'mape'])
  return model


def getModelInterpolator(lr=0.001):
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


def fromCsv(csv):
  ts = pd.to_datetime(csv["timestamp"], format="%d/%m/%Y %H:%M").apply(lambda x: x.hour).to_numpy()
  csv = csv.drop(["timestamp"], axis=1)
  x = csv.to_numpy()
  m = np.any(np.isnan(x), axis=1).astype(int)
  m = np.concatenate(([0], np.cumsum(m)))
  idx = np.argwhere((m[24:] - m[:-24]) == 0).flatten()
  if len(idx) == 0:
    return None, 97
  idx = idx[-1]
  z = np.zeros((24, 3))
  x = np.concatenate((x, z), axis=0)[idx:idx+47]
  tt = (np.ogrid[0:47] + ts[idx]) % 24
  tt_sin = np.expand_dims(np.sin(tt*np.pi/12), axis=1)
  tt_cos = np.expand_dims(-np.cos(tt*np.pi/12), axis=1)
  tt_id = np.ones_like(tt_sin)
  tt_id[:24] = -1
  x = np.concatenate((x, tt_sin, tt_cos, tt_id), axis=1)
  return x, np.max((idx, 97))


def fromForecasterResult(y_pred, idx):
  x = []
  for i in range(len(idx)):
    y = y_pred[i][-24:,0:1]
    if idx[i] < 0:
      y = np.roll(y, idx[i], axis=0)
      y[idx[i]:] = np.nan
    x.append(y)
  x.append(np.zeros_like(x[0]))
  x.append(np.zeros_like(x[0]))
  return np.concatenate(x, axis=1)


def generateSingleOutput(forecaster, extrapolator, folder, drop=[]):
  X_fr = []
  idx = []
  for i in range(11):
    if i in drop:
      continue
    path = f"{folder}/{stations[i]}"
    data = pd.read_csv(path, index_col=0)
    x, id = fromCsv(data)
    if x is None:
      x = np.zeros((47, 6))
    X_fr.append(x)
    idx.append(id-144)
  X_fr = np.array(X_fr)
  X_fr[:,:24,:3] = (X_fr[:,:24,:3] - para["mean"])/para["std"]
  y_pred = forecaster.predict(X_fr, verbose=0)
  X_it = fromForecasterResult(y_pred, idx)
  X_it = np.nan_to_num(X_it, nan=-1)

  output = []
  for coord in coords:
    X_it[:,-2:] = coord
    output.append(extrapolator.predict(X_it, verbose=0))
  idx = np.array(idx)
  return output, np.mean(np.max((idx, -np.ones_like(idx)*24), axis=0))


def main(args):
  drop = []
  p = 1
  for i in range(11):
    if (args.drop_mask & p) != 0:
      drop.append(i)
    p *= 2

  global stations, coords, para

  forecaster = getModelForecaster()
  extrapolator = getModelInterpolator()

  forecaster.load_weights("weights/forecaster.h5")
  extrapolator.load_weights("weights/extrapolator.h5")

  parentFolder = f"{args.test_path}/input/%d"
  if os.path.exists(args.output_path):
    shutil.rmtree(args.output_path)
  os.mkdir(args.output_path)

  coords = 10*(pd.read_csv(f"{args.test_path}/location.csv", \
                           usecols=["longitude", "latitude"]).to_numpy() - np.array([105.8, 21]))
  para = np.load("paras/forecaster.npz")
  stations = os.listdir(f"{args.test_path}/input/1")

  print(coords)
  for i in tqdm(range(100)):
    y, _ = generateSingleOutput(forecaster, extrapolator, parentFolder % (i+1), drop)
    os.mkdir(f"{args.output_path}/{i+1}")
    for j in tf.range(4):
      outputFile = f"{args.output_path}/{i+1}/res_{i+1}_{j+1}.csv"
      out = pd.DataFrame()
      out["PM2.5"] = y[j].flatten()
      out.to_csv(outputFile, index=False)

  if os.path.exists("prediction.zip"):
    shutil.remove("prediction.zip")

  shutil.make_archive("prediction", "zip", args.output_path)
  shutil.rmtree(args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--test-path", type=str, default="./data/public-test", help="Path of the testing data folder (default: public-test folder in the same location of this file)")
  parser.add_argument("--output-path", type=str, default="result", help="Path to the temporary output folder (do not make it `.`, `..` or something like that. You know what happen) (default: result/)")
  parser.add_argument("--drop-mask", type=int, default=0, help="(Advanced) Bitmask indicating which stations have been excluded from data during the training of the extrapolator. (default: 0 - nomask)")
  args = parser.parse_args()
  main(args)