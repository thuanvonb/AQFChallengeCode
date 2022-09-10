import os
import sys
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

para = None

def getModelForecaster(lr=0.001, wl_sample_weight=0.008):
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
  model.compile(optimizer=Adam(learning_rate=lr), loss="huber", metrics=['mae', 'mape'])
  return model


def getModelExtrapolator(lr=0.001):
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
  return model


def fromCsv(csv, meteo):
  ts = pd.to_datetime(csv["time"], format="%Y-%m-%d %H:%M:%S").apply(lambda x: x.hour).to_numpy()
  csv = csv.drop(["time"], axis=1)
  x = csv.to_numpy()
  m = np.any(np.isnan(x), axis=1).astype(int)
  m = np.concatenate(([0], np.cumsum(m)))
  idx = np.argwhere((m[24:] - m[:-24]) == 0).flatten()
  if len(idx) == 0:
    return None, 97
  idx = idx[-1]
  z = np.zeros((24, 8))
  x = np.concatenate((meteo, x), axis=1)
  x = np.concatenate((x, z), axis=0)[idx:idx+47]
  tt = (np.ogrid[0:47] + ts[idx]) % 24
  tt_sin = np.expand_dims(np.sin(tt*np.pi/12), axis=1)
  tt_cos = np.expand_dims(-np.cos(tt*np.pi/12), axis=1)
  tt_id = np.ones_like(tt_sin)
  tt_id[:24] = -1
  x = np.concatenate((x, tt_sin, tt_cos, tt_id), axis=1)
  return x, np.max((idx, 97))


def fromForecasterResult(y_pred, idx, coords):
  X = np.zeros((24, 26))
  for i in range(len(idx)):
    y = y_pred[i][-24:,0:1]
    if idx[i] < 0:
      y = np.roll(y, idx[i], axis=0)
      y[idx[i]:] = np.nan
    X[:, i*3:i*3+1] = y
    X[:, i*3+1:i*3+3] = coords[i]
  return X


def generateSingleOutput(forecaster, extrapolator, folder):
  X_fr = []
  idx = []
  stations = os.listdir(f"{folder}/")
  stations = list(filter(lambda st: st.startswith("S000"), stations))
  stations.sort(key=lambda st: st[:8])
  input_coords = pd.read_csv(f"{folder}/location_input.csv")
  input_coords.sort_values(by='location', key=lambda st: st[:8])
  input_coords = input_coords.drop(["location"], axis=1)
  input_coords = input_coords.to_numpy()
  input_coords -= np.array([105.8, 21])
  meteo = pd.read_csv(f"{folder}/meteo/station_86.csv").drop(["time"], axis=1).to_numpy()
  x = []
  for i in range(3):
    k = meteo[:-1]*(3-i)/3 + meteo[1:]*i/3
    x.append(np.expand_dims(k, axis=1))
  meteo = np.concatenate(x, axis=1).reshape((-1, 5))
  for i in range(len(stations)):
    path = f"{folder}/{stations[i]}"
    data = pd.read_csv(path, index_col=0)
    x, id = fromCsv(data, meteo)
    if x is None:
      x = np.zeros((47, 11))
    X_fr.append(x)
    idx.append(id-144)
  X_fr = np.array(X_fr)
  X_fr[:,:24,:-3] = (X_fr[:,:24,:-3] - para["mean"])/para["std"]
  # print(X_fr[0])
  y_pred = forecaster.predict_on_batch(X_fr)
  X_it = fromForecasterResult(y_pred, idx, input_coords)
  X_it = np.nan_to_num(X_it, nan=-1)
  # print(X_it)

  output = []
  output_coords = pd.read_csv(f"{folder}/location_output.csv").drop(["location"], axis=1)
  output_coords = output_coords.to_numpy() - np.array([105.8, 21])
  for coord in output_coords:
    X_it[:,-2:] = coord
    output.append(extrapolator.predict(X_it, verbose=0))
  idx = np.array(idx)
  return output, np.mean(np.max((idx, -np.ones_like(idx)*24), axis=0))


def main(args):
  print("Generating output...")
  global para

  forecaster = getModelForecaster()
  extrapolator = getModelExtrapolator()

  forecaster.load_weights("weights/forecaster.h5")
  extrapolator.load_weights("weights/extrapolator.h5")

  parentFolder = f"{args.test_path}/input/%d"
  if os.path.exists(args.output_path):
    shutil.rmtree(args.output_path)
  os.mkdir(args.output_path)

  para = np.load("paras/forecaster.npz")

  for i in tqdm(range(89)):
    y, _ = generateSingleOutput(forecaster, extrapolator, parentFolder % (i+1))
    os.mkdir(f"{args.output_path}/{i+1}")
    for j in tf.range(6):
      outputFile = f"{args.output_path}/{i+1}/res_{i+1}_{j+1}.csv"
      out = pd.DataFrame()
      out["PM2.5"] = y[j].flatten()
      out.to_csv(outputFile, index=False)

  if os.path.exists("prediction.zip"):
    os.remove("prediction.zip")

  shutil.make_archive("prediction", "zip", args.output_path)
  shutil.rmtree(args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--test-path", type=str, default="./test", help="Path of the testing data folder (default: ./test)")
  parser.add_argument("--output-path", type=str, default="result", help="Path to the temporary output folder (do not make it `.`, `..` or something like that. You know what would happen) (default: result/)")
  args = parser.parse_args()
  main(args)