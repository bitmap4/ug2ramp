import torch
from models import LinearRegression, LogisticRegression
from torch.utils.data import DataLoader
from datasets import load_dataset

# DS_NAME = "clf_cat_albert"
DS_NAME = "reg_cat_SGEMM_GPU_kernel_performance"
# TYPE = "LogisticRegression" if DS_NAME.startswith("clf") else "LinearRegression"
TYPE = "LinearRegression"

# Load dataset
ds = load_dataset("inria-soda/tabular-benchmark", DS_NAME)["train"]
train_size = int(0.9 * len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False)

class_names = ds.column_names[-1:]

# Calculate normalization parameters from training data
print("Calculating normalization parameters...")
all_inputs = []
for batch in train_dataloader:
    inputs = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k not in class_names], dim=1)
    all_inputs.append(inputs)

all_inputs = torch.cat(all_inputs, dim=0)
input_mean = all_inputs.mean(dim=0)
input_std = all_inputs.std(dim=0)
# Add small epsilon to prevent division by zero
input_std = torch.where(input_std < 1e-6, torch.ones_like(input_std), input_std)

print(f"Input mean range: {input_mean.min():.4f} to {input_mean.max():.4f}")
print(f"Input std range: {input_std.min():.4f} to {input_std.max():.4f}")

# Training loop
predictor = LinearRegression(n=len(ds.features)) if TYPE == "LinearRegression" \
            else LogisticRegression(n=len(ds.features))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(predictor.parameters(), lr=0.0002)

for epoch in range(6):
    total_loss = 0
    for batch in train_dataloader:
        inputs = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k not in class_names], dim=1)
        inputs = (inputs - input_mean) / input_std
        targets = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k in class_names], dim=1)
        
        # Forward pass
        outputs = predictor(inputs)
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}")

# Evaluate the model
with torch.no_grad():
    total_loss = 0
    for batch in test_dataloader:
        inputs = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k not in class_names], dim=1)
        inputs = (inputs - input_mean) / input_std
        targets = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k in class_names], dim=1)

        outputs = predictor(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss}")

# show test prediction examples
print("First 5 test predictions:")
with torch.no_grad():
    for batch in test_dataloader:
        inputs = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k not in class_names], dim=1)
        inputs = (inputs - input_mean) / input_std
        targets = torch.stack([v.detach().clone().to(torch.float32) for k, v in batch.items() if k in class_names], dim=1)

        outputs = predictor(inputs)
        predictions = outputs.numpy()
        print("Targets:", targets.numpy()[:5])
        print("Predictions:", predictions[:5])
        break