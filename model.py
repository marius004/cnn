from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
from torchvision import transforms
import torch.nn.functional as F
from skimage import io
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os

DATASET_DIR = 'data'
BATCH_SIZE = 128
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 25
NUM_CLASSES = 3
INPUT_CHANNELS = 3
IMAGE_SIZE = (80, 80, 3)

def load_labels(file_name):
    df = pd.read_csv(os.path.join(DATASET_DIR, file_name))
    return dict(zip(df['image_id'], df['label']))

def preprocess_image(image_path):
    image = io.imread(image_path)
    return resize(image, IMAGE_SIZE, anti_aliasing=True).astype(np.float32)

def preprocess_images(image_ids, labels, dataset_type):
    X, y = [], []
    for image_id in image_ids:
        image_path = os.path.join(DATASET_DIR, dataset_type, image_id + '.png')
        X.append(preprocess_image(image_path))
        y.append(labels[image_id])
    return np.array(X), np.array(y)

class CustomDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        if self.y is not None:
            label = self.y[idx]
            return image, label
        return image

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        l2_regularization = sum(torch.norm(param)**2 for param in self.parameters())
        loss += self.weight_decay * l2_regularization
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, "
              f"train_acc: {result['train_acc']:.4f}, val_loss: {result['val_loss']:.4f}, "
              f"val_acc: {result['val_acc']:.4f}")

class Model(ImageClassificationBase):
    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, file_name='model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = file_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(model)
        elif val_acc >= self.best_score:
            self.best_score = val_acc
            self.save_checkpoint(model)
            self.counter = 0
        else:
            if val_acc < self.best_score - self.delta:
                self.counter += 1
                self.early_stop = (self.counter >= self.patience)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def setup_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, early_stopping=None):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        correct = 0
        total = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        result = evaluate(model, val_loader)
        result['train_loss'] = np.mean(train_losses)
        result['train_acc'] = train_acc
        model.epoch_end(epoch, result)
        history.append(result)

        val_acc = result['val_acc']
        if early_stopping:
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return history

def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def load_and_preprocess_labels():
    train_labels = load_labels('train.csv')
    validation_labels = load_labels('validation.csv')
    return train_labels, validation_labels

def load_and_preprocess_images(train_labels, validation_labels):
    train_df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(DATASET_DIR, 'validation.csv'))
    test_images = os.listdir(os.path.join(DATASET_DIR, 'test'))

    X_train, y_train = preprocess_images(train_df['image_id'], train_labels, 'train')
    X_validation, y_validation = preprocess_images(validation_df['image_id'], validation_labels, 'validation')
    X_test = [preprocess_image(os.path.join(DATASET_DIR, 'test', img)) for img in test_images]
    
    return X_train, y_train, X_validation, y_validation, X_test, test_images

def initialize_model():
    model = Model(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(device)
    early_stopping = EarlyStopping(patience=PATIENCE)
    return model, early_stopping

def setup_dataloaders(X_train, y_train, X_validation, y_validation, X_test):
    data_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    data_transform_validation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=data_transform_train)
    validation_dataset = CustomDataset(X_validation, y_validation, transform=data_transform_validation)
    test_dataset = CustomDataset(X_test, transform=data_transform_validation)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation_dataset, BATCH_SIZE * 2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE * 2, num_workers=4, pin_memory=True)
    
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, early_stopping):
    history = fit(NUM_EPOCHS, LEARNING_RATE, model, train_loader, val_loader, torch.optim.Adam, early_stopping)
    return history

def make_predictions(model, test_loader, test_images):
    predictions = get_predictions(model.to('cpu'), test_loader)
    predictions_df = pd.DataFrame({'image_id': test_images, 'label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions for test data saved to predictions.csv")

if __name__ == "__main__":
    train_labels, validation_labels = load_and_preprocess_labels()
    X_train, y_train, X_validation, y_validation, X_test, test_images = load_and_preprocess_images(train_labels, validation_labels)
    
    device = setup_device()
    model, early_stopping = initialize_model()
    
    train_loader, val_loader, test_loader = setup_dataloaders(X_train, y_train, X_validation, y_validation, X_test)
    
    train_model(model, train_loader, val_loader, early_stopping)

    best_model = Model(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(torch.load(early_stopping.path))
    
    make_predictions(best_model, test_loader, test_images)