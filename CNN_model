import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class test_nn (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 16, 3) # in_channels, out_channels, kernel_size
        self.layer2 = nn.Conv2d(16, 32, 3) # in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2) # pour réduire la taille
        self.fc = nn.Linear(800, 1) # pour la classification finale

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x) # Active
        x = self.pool(x) # Pooling
        x = self.layer2(x)
        x = torch.relu(x) # Active
        x = self.pool(x) # Pooling
        x = x.flatten(1) # Flatten
        x = self.fc(x)
        return x
    
model = test_nn()
x = torch.randn(1, 1, 28, 28)
sortie = model(x)
print(sortie.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss() # pour la classification binaire


class MonDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

image  = torch.randn(50, 1, 28, 28) # 50 images de 28x28
labels = torch.randint(0, 2, (50, 1)).float()

data = MonDataset(image, labels.float())  # ← créer le Dataset d'abord
dataloader = DataLoader(data, batch_size=10, shuffle=True)


for epoch in range (100):
    for x_batch,y_batch in dataloader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred,y_batch)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}")