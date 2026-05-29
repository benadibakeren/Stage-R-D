import torch
import torch.nn as nn

class simple_nn (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x) # Active
        x = self.layer2(x)
        return x
    
model = simple_nn()
y = torch.randint(0, 2, (5, 1)).float()
x = torch.randn(5, 2)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

# Training loop
for epoch in range(50): 
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred,y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Evaluation loop
x_test = torch.randn(1,2)
y_test = torch.randint(0, 2, (1, 1)).float()


with torch.no_grad(): #don't keep grad, just evaluate
    pred_test = model(x_test)
    print(f"Test Input: {x_test}, Test Output: {pred_test}, Test Label: {y_test}")
    loss_test = criterion(pred_test, y_test)
    print(f"Test Loss: {loss_test.item():.4f}")

