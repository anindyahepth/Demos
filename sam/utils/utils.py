import os
import mlflow
import torch

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

##############DATASET##############################

class CifarDataset(Dataset):
    def __init__(self, dataset, augment = True):
        self.dataset = dataset
        self.augment = augment

        # Preprocess the image
        if self.augment:
          self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.247, 0.243, 0.261]
                )
            ])

        else:
          self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        label = item['label']

        pixel_values = self.transform(image)
        labels = torch.tensor(label)

        dataset_dict = {"pixel_values": pixel_values.squeeze(), "labels": labels}

        return dataset_dict
##############################################
###DATALOADERS
##############################################

def get_datasets(augment=False):
  from datasets import load_dataset
  dataset = load_dataset("uoft-cs/cifar10")

  dataset_train = CifarDataset(dataset['train'], augment=augment)
  dataset_test = CifarDataset(dataset['test'], augment=False)

  return dataset_train, dataset_test

def get_dataloaders(dataset_train, dataset_test):

  from torch.utils.data import DataLoader

  num_workers = 4

  torch.manual_seed(123)

  def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

  train_loader = DataLoader(
    dataset=dataset_train,
    batch_size=256,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    persistent_workers= True,
    drop_last=True,
)

  test_loader = DataLoader(
    dataset=dataset_test,
    batch_size=28,
    collate_fn=collate_fn,
    num_workers=num_workers,
    persistent_workers= True,
    drop_last=False,
)
  return train_loader, test_loader
##############################################
###DISABLE_BN_STATS_FUNC
##############################################
def disable_bn_stats(model):
    """
    Disable running stats tracking for all BatchNorm variants.
    Works with standard PyTorch BatchNorm and timm's BatchNormAct layers.
    """
    for module in model.modules():
        # Standard PyTorch BatchNorm
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
        # timm's BatchNormAct variants (check by class name)
        elif 'BatchNormAct' in module.__class__.__name__:
            module.track_running_stats = False

def enable_bn_stats(model):
    """
    Enable running stats tracking for all BatchNorm variants.
    Works with standard PyTorch BatchNorm and timm's BatchNormAct layers.
    """
    for module in model.modules():
        # Standard PyTorch BatchNorm
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = True
        # timm's BatchNormAct variants (check by class name)
        elif 'BatchNormAct' in module.__class__.__name__:
            module.track_running_stats = True
##############################################

##############################################
def get_model_resnet():
  import timm
  model = timm.create_model('resnetv2_18', pretrained=False, num_classes=10)

  return model
##############################################
#Training_Function
##############################################
# trains model for one epoch and computes training loss and training accuracy

def train_single_epoch(model, train_loader, device, optimizer, epoch, criterion = nn.CrossEntropyLoss()):


    use_sam = hasattr(optimizer, 'first_step') and hasattr(optimizer, 'second_step')


    model.train()
    train_loss = 0.0
    correct =0
    N_total =0

    for i, batch in enumerate(train_loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        if use_sam:
          enable_bn_stats(model)
          output = model(pixel_values)
          loss = criterion(output, labels)
          loss.backward()
          optimizer.first_step(zero_grad=True)

          disable_bn_stats(model)
          output = model(pixel_values)
          loss = criterion(output, labels)
          loss.backward()
          enable_bn_stats(model)
          optimizer.second_step(zero_grad=True)

        else:
          output = model(pixel_values)
          loss = criterion(output, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)

    model.eval()
    with torch.no_grad():
      for batch in train_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        output_for_acc = model(pixel_values)
        _, label_preds = torch.max(output_for_acc, dim=1)
        correct += (label_preds == labels).sum().item()
        N_total += labels.size(0)

    train_acc = correct/N_total

    print(f"Training loss after epoch {epoch}:", train_loss)
    print(f"Training acc after epoch {epoch}:", train_acc)
    #print("-" * 60)


    return train_loss, train_acc
 ##############################################
###INFERENCE ON THE TEST DATASET
 ##############################################
def inference(model, dataloader, device):
  model.eval()

  correct = 0
  N_tot = 0

  with torch.no_grad():
    for i,batch in enumerate(dataloader):
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["labels"].to(device)

      output = model(pixel_values)
      _, label_preds = torch.max(output, dim=1)
      correct += (label_preds == labels).sum().item()
      N_tot += labels.size(0)

    test_acc = correct/N_tot
    print(f"Test acc:", test_acc)
    print("-" * 60)

  return test_acc
