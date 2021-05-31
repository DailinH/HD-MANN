import torch
import torch.nn as nn
import glob
import os
import sys
import numpy as np 
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from more_itertools import unzip

from cnn import CNNController
from data_generator import DataGenerator
from utils import *

def train(model, data_generator, optimizer, criterion, device, n_step = 50000, save=True):
    
  exp_name = f"{W}way{S}shot"
  writer = SummaryWriter(comment=exp_name)
  
  loss_train = []
  val_accs = []
  steps = []
  for step in range(n_step):
    #prep data
    support_label, support_set, query_label, query_set = data_generator.sample_batch('train', 32)
    support_label, support_set = prep_data(support_label, support_set, device)
    query_label, query_set = prep_data(query_label, query_set, device)
    #support set loading
    support_keys = None
    model.eval()
    with torch.no_grad():
      support_keys = model(support_set)

    #query evaluation
    model.train()
    query_keys = model(query_set)
    cosine_sim = get_cosine_similarity(query_keys, support_keys)
    # sharpened = sharpening_softabs(cosine_sim, 10)
    if sharpen == 'softmax':
      print("Using softmax sharpening function")
      sharpened = sharpening_softmax(cosine_sim)
    elif sharpen == 'Using softabs sharpening function':
      sharpened = sharpening_softabs(cosine_sim, 10)
    normalized = normalize(sharpened)
    pred = weighted_sum(normalized, support_label)
    optimizer.zero_grad()
    loss = criterion(pred, query_label)
    loss.backward()
    if step % 500 == 0:
      print(f"train loss = {loss}")
      writer.add_scalar("Loss/Train", loss, step)
      acc = inference(model, data_gen, device)
      print(f"val acc = {acc}")
      loss_train.append(loss)
      val_accs.append(acc)
      steps.append(step)
    #backprop
    optimizer.step()
    
  if save:
      torch.save(model.state_dict(), f"model_{exp_name}")
  
  plt.plot(steps,loss_train)
  plt.xlabel("iteration")
  plt.ylabel('training loss')
  plt.show()

  plt.plot(steps,val_accs)
  plt.xlabel("iteration")
  plt.ylabel('validation accuracy')
  plt.show()
  return model


def inference(model, data_generator, device, key_mem_transform = binarize, n_step = 1000):
  model.eval()
  accumulated_acc = 0
  for i in range(n_step):
    support_label, support_set, query_label, query_set = data_generator.sample_batch('val', 32)
    support_label, support_set = prep_data(support_label, support_set, device)
    query_label, query_set = prep_data(query_label, query_set, device)
    support_label = support_label.cpu().numpy()
    query_label = query_label.cpu().numpy()
    with torch.no_grad():
      support_keys = key_mem_transform(model(support_set).cpu().detach().numpy())
      query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
      dot_sim = get_dot_prod_similarity(query_keys, support_keys)
      sharpened = np.abs(dot_sim)
      pred = np.dot(sharpened, support_label)
      pred_argmax = np.argmax(pred, axis = 1)
      query_label_argmax = np.argmax(query_label, axis = 1)
      # print(np.sum(pred_argmax == query_label_argmax))
      accumulated_acc += np.sum(pred_argmax == query_label_argmax)/len(pred_argmax)
  return accumulated_acc/n_step



device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device('cuda')

W = 5 #way
S = 1 #shots
D = 512
data_gen = DataGenerator(W,S)
model = CNNController(D).float().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

model = train(model, data_gen, optimizer, criterion, device, 50000)
torch.save(model.state_dict(), './model/')

acc = inference(model, data_gen, device)
print(f"acc = {acc}")

