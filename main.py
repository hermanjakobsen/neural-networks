import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

from model import Net
from training import train

# We set a fixed seed for repeatability
random_seed = 12345  # This seed is also used in the pandas sample() method below
torch.manual_seed(random_seed)

df = pd.read_csv('data/well_data.csv', index_col=0)

# # Plot the features

fig, ax = plt.subplots(2, 2, figsize=(16, 9))

# Choke valve opening
ax[0, 0].plot(df['CHK'], label='CHK')
ax[0, 0].legend()

# Total flow through choke valve
ax[0, 1].plot(df['TWH'], label='TWH')
ax[0, 1].legend()

# Diff pressure over choke valve
ax[1, 0].plot(df['PWH'] - df['PDC'], label='PWH - PDC')
ax[1, 0].legend()

# Fractions
ax[1, 1].plot(df['FOIL'], label='FOIL')
ax[1, 1].plot(df['FGAS'], '--r', label='FGAS')
ax[1, 1].legend()



# Test set (this is the period for which we must estimate QTOT)
test_set = df.iloc[2000:2500]

# Make a copy of the dataset and remove the test data
train_val_set = df.copy().drop(test_set.index)

# Sample validation data without replacement (10%)
val_set = train_val_set.sample(frac=0.1, replace=False, random_state=random_seed)

# The remaining data is used for training (90%)
train_set = train_val_set.copy().drop(val_set.index)

# Check that the numbers add up
n_points = len(train_set) + len(val_set) + len(test_set)
print(f'{len(df)} = {len(train_set)} + {len(val_set)} + {len(test_set)} = {n_points}')


# # Plot the train, validation and test set

plt.figure(figsize=(16, 9))
plt.scatter(train_set.index, train_set['QTOT'], color='black', label='Train')
plt.scatter(val_set.index, val_set['QTOT'], color='green', label='Val')
plt.scatter(test_set.index, test_set['QTOT'], color='red', label='Test')
plt.legend()
#plt.show()

# # Prepare data for training
# 
# We use the PyTorch DataLoader to simplify the partitioning of the training data into batches.

# Define the target and features
INPUT_COLS = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
OUTPUT_COLS = ['QTOT']

# Get input and output tensors and convert them to torch tensors
x_train = torch.from_numpy(train_set[INPUT_COLS].values).to(torch.float)
y_train = torch.from_numpy(train_set[OUTPUT_COLS].values).to(torch.float)

x_val = torch.from_numpy(val_set[INPUT_COLS].values).to(torch.float)
y_val = torch.from_numpy(val_set[OUTPUT_COLS].values).to(torch.float)

# Create dataset loaders
# Here we specify the batch size and if the data should be shuffled
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_set), shuffle=False)


# # Construct and initialize the model
# 
# The time to construct an actual neural network has finally come! 
# 
# Now, what should the network size be? How much data do we have? How nonlinear is the underlying function we are trying to model? Two to hidden layers with 50 units may be sufficient? Let's try that.


layers = [len(INPUT_COLS), 50, 50, len(OUTPUT_COLS)]
net = Net(layers)

print(f'Layers: {layers}')
print(f'Number of model parameters: {net.get_num_parameters()}')
# print(6*50 + 50 + 50*50 + 50 + 50 * 1 + 1)


# # Train the model
# 
# Almost there. We only need to set some important hyper-parameters before we start the training. The number of epochs to train, the learning rate, and the L2 regularization factor.

n_epochs = 300
lr = 0.001
l2_reg = 0.001  # 10

log_dir = 'logs'
log_iter = 100
writer = None # SummaryWriter(log_dir)
net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg, log_iter, writer)


# # Evaluate the model on validation data


# Predict on validation data
pred_val = net(x_val)

# Compute MSE, MAE and MAPE on validation data
print('Error on validation data')

mse_val = torch.mean(torch.pow(pred_val - y_val, 2))
print(f'MSE: {mse_val.item()}')

mae_val = torch.mean(torch.abs(pred_val - y_val))
print(f'MAE: {mae_val.item()}')

mape_val = 100*torch.mean(torch.abs(torch.div(pred_val - y_val, y_val)))
print(f'MAPE: {mape_val.item()} %')


# # Are you happy with the result? 
# 
# Remember that the validation error is just an estimate of the test error (the error on new examples). The test error may be higher or lower. We may proceed if we believe that we have a good validation set and the validation error is sufficiently low.
# 
# If we wish to proceed, a final step before we produce our estimates would be to re-train the model on both the train and validation data. We will skip this step for now.
# 
# Let's see how well we did!

# # Evaluate the model on test data


# Get input and output as torch tensors
x_test = torch.from_numpy(test_set[INPUT_COLS].values).to(torch.float)
y_test = torch.from_numpy(test_set[OUTPUT_COLS].values).to(torch.float)

# Make prediction
pred_test = net(x_test)

# Compute MSE, MAE and MAPE on test data
print('Error on test data')

mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
print(f'MSE: {mse_test.item()}')

mae_test = torch.mean(torch.abs(pred_test - y_test))
print(f'MAE: {mae_test.item()}')

mape_test = 100*torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
print(f'MAPE: {mape_test.item()} %')


plt.figure(figsize=(16, 9))
plt.plot(y_test.numpy(), label='Missing QTOT')
plt.plot(pred_test.detach().numpy(), label='Estimated QTOT')
plt.legend()
#plt.show()