from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n, d, h=1, c=6):
        super().__init__()
        self.U = torch.nn.Embedding(n, h)
        self.V = torch.nn.Embedding(d, h)
        self.classfier = torch.nn.Linear(2 * h, c)

    def forward(self, X_batch):
        U = self.U(X_batch[:, 0])
        V = self.V(X_batch[:, 1])
        out = torch.cat((U, V), 1)
        return self.classfier(out)


n, d, h = 8, 8, 2
W = torch.tensor(
    [
        [1, 1, 0, 1, 1, 1, 5, 5],
        [1, 0, 1, 1, 0, 1, 4, 5],
        [1, 1, 1, 0, 1, 1, 5, 5],
        [1, 1, 1, 1, 0, 1, 4, 5],
        [1, 0, 1, 1, 1, 1, 5, 3],
        [0, 1, 0, 1, 0, 1, 5, 3],
        [0, 1, 0, 0, 1, 1, 5, 3],
        [0, 1, 1, 1, 0, 1, 5, 3],
    ],
    dtype=torch.float32,
)


data = np.array([(i, j, W[i, j]) for i in range(n) for j in range(d)])


# defining the Dataset class
class data_set(Dataset):
    def __init__(self, data):
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = data_set(data)
# implementing dataloader on the dataset and printing per batch
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)


# Define training parameters
LR = 1e-1
EPOCHS = 1000
model = MatrixFactorization(n, d, h)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Relevant if you have a GPU:
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
W = W.to(device)


running_loss = []
for epoch in tqdm(range(EPOCHS)):
    epoch_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        optimizer.zero_grad()
        # X_batch is (i,j) index must be int
        X_batch = batch_X.to(torch.int).to(device)
        # y_batch must be float
        y_batch = batch_y.to(torch.long).to(device)
        y_pred = model(X_batch)
        loss = loss_func(y_pred, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(dataloader)
    running_loss.append(epoch_loss)

plt.plot(running_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


# Données de test
W_test = torch.tensor(
    [
        [3, 1, 0, 1, 1, 1, 5, 5],
        [1, 0, 1, 4, 0, 0, 4, 5],
        [1, 1, 1, 0, 1, 1, 5, 5],
        [1, 2, 1, 1, 0, 1, 4, 5],
        [1, 0, 1, 3, 4, 1, 5, 5],
        [0, 1, 0, 1, 0, 1, 2, 3],
        [0, 1, 0, 0, 1, 1, 5, 3],
        [0, 3, 1, 1, 0, 1, 5, 3],
    ],
    dtype=torch.float32,
)


data_test = np.array([(i, j, W_test[i, j]) for i in range(n) for j in range(d)])
dataset_test = data_set(data_test)
dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

# Predicting the ratings
with torch.no_grad():
    predictions = (
        model(torch.tensor(data_test[:, :2], dtype=torch.int).to(device)).cpu().numpy()
    )
# print(predictions.shape)
# print(predictions)

# Calculating the accuracy
y_test = data_test[:, 2].astype(int)
y_pred = np.argmax(predictions, axis=1)
accuracy = np.mean(y_test == y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculating the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix : ", confusion_matrix)

# Plotting the accuracy with linking the points
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label="True Ratings", marker="o")
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Ratings", marker="x")

# Linking the points
for i in range(len(y_test)):
    plt.plot([i, i], [y_test[i], y_pred[i]], "r--")

plt.xlabel("Sample Index")
plt.ylabel("Rating")
plt.title("True vs Predicted Ratings")
plt.legend()
plt.show()
