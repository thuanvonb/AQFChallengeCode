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
    np.random.seed(31415)


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
  files = os.listdir(f"{trainDataFolder}/air")
  filesId = [name[:8] for name in files]
  stations = []
  coords_input = []
  for i in range(len(loc_input)):
    id_ = loc_input['location'][i][:8]
    stations.append(pd.read_csv(f"{trainDataFolder}/air/{files[filesId.index(id_)]}", index_col=0))
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
  return model


def main(args):
  print("Extrapolator operating...")
  print("--------------------------------------------------------")
  print("Initializing data...")
  datapieces, coords = generateData(args.train_path, drop=[])
  trainGen = Generator(datapieces, coords, args.training_sample, args.batch_size)
  valGen = Generator(datapieces, coords, 100, args.batch_size)
  testGen = Generator(datapieces, coords, 100, args.batch_size)
  print("Training...")
  model = getModel(lr=args.learning_rate)
  model.fit(trainGen, epochs=1, validation_data=valGen, validation_steps=len(valGen))

  if not os.path.exists("weights"):
    os.mkdir("weights")

  model.save_weights("weights/extrapolator.h5")
  print("Weights saved in weights/extrapolator.h5")
  print("Extrapolator terminated\n")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-path", type=str, default="./train", help="Path of the training data folder (default: ./train)")
  parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate of the forecaster (default: 0.001)")
  parser.add_argument("--training-sample", type=int, default=300000, help="Number of sample to train the model (default: 300000)")
  parser.add_argument("--batch-size", type=int, default=128, help="Batch size for a single training step (default: 128)")
  args = parser.parse_args()
  main(args)
  