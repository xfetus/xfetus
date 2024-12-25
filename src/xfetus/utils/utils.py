import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             precision_recall_fscore_support)


def save_model(model, optimizer, epoch, best_epoch, train_losses, val_losses, save_path):
  state = {
    'epoch': epoch,
    'best_epoch': best_epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses
  }
  torch.save(state, save_path)

def load_model(model_path, device, model=None, optimizer=None):
  if device == "cpu":
    state = torch.load(model_path, map_location=torch.device('cpu'))
  else:
    state = torch.load(model_path)

  epoch = state['epoch']
  best_epoch = state['best_epoch']
  train_losses = state['train_losses']
  val_losses = state['val_losses']

  if model:
    model.load_state_dict(state['state_dict'])
  if optimizer:
    optimizer.load_state_dict(state['optimizer'])

  return epoch, best_epoch, model, optimizer, train_losses, val_losses

def train_model(train_dataloader,
                val_dataloader,
                model,
                optimizer,
                criterion,
                num_epochs,
                model_save_path,
                checkpoint_save_path,
                device,
                load_checkpoint=False):
  start = time.time()
  train_losses = []
  val_losses = []
  best_loss = 100
  best_epoch = 0
  s_epoch = 0

  if load_checkpoint:
    s_epoch, best_epoch, model, optimizer, train_losses, val_losses = load_model(checkpoint_save_path, device, model, optimizer)
    best_loss = val_losses[best_epoch]

  #TODO add if to print or not model
  # print(model)
  for epoch in range(s_epoch, num_epochs):
    model.train()
    running_loss = 0
    for i, (inputs, labels) in enumerate(train_dataloader):
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels).mean()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if i % 10 == 0:
        print(f"Epoch {epoch} iteration {i} training loss: {loss.item()}")

    train_epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(train_epoch_loss)


    model.eval()
    with torch.no_grad():
      running_loss = 0
      for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels).mean()
        running_loss += loss.item()
      val_epoch_loss = running_loss / len(val_dataloader)
      val_losses.append(val_epoch_loss)
      if epoch == 0:
        best_loss = val_epoch_loss
      elif val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        best_epoch = epoch
        save_model(model, optimizer, epoch, best_epoch, train_losses, val_losses, model_save_path)

      save_model(model, optimizer, epoch, best_epoch, train_losses, val_losses, checkpoint_save_path)
      print(f"Epoch {epoch} average training loss: {train_epoch_loss}, average validation loss: {val_epoch_loss}")
      end = time.time()
      print(f"Running time: {end - start}s")
  end = time.time()
  print(f"Execution time: {end - start}s")


def plot_training_curve(train_losses, val_losses, best_epoch=None, title=None):
  plt.plot(train_losses, label="Train")
  plt.plot(val_losses, label="Test")
  if title:
    plt.title(title)
  if best_epoch:
    plt.axvline(x=best_epoch, color='#9f9f9f', label=(f"Best Epoch: {best_epoch}"), ls="--", lw=1)
  plt.legend()
  plt.show()

def evaluate_model(y_true, y_pred, class_names):
  accuracy = accuracy_score(y_true, y_pred)
  p, r, f, s = precision_recall_fscore_support(y_true, y_pred)

  print(f"Accuracy: {accuracy}")
  print(f"Precision: {p}")
  print(f"Recall: {r}")
  print(f"F1: {f}")
  print(f"Support: {s}")

  ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
  plt.show()
