import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        lr: float = 0.001,
        l2_reg: float = 0,
        log_iter: int = 100,
        writer: SummaryWriter = None,
) -> torch.nn.Module:
    """
    Train model using mini-batch SGD
    After each epoch, we evaluate the model on validation data

    :param net: initialized neural network
    :param train_loader: DataLoader containing training set
    :param n_epochs: number of epochs to train (default: 100)
    :param lr: learning rate (default: 0.001)
    :param l2_reg: L2 regularization factor (default: 0)
    :param log_iter: how often the logger should write (default: 100)
    :param writer: logger (default: None)
    :return: torch.nn.Module: trained model.
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    global_step = 0
    # Train Network
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            global_step += 1

            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            
            # Compute cost function
            batch_mse = criterion(outputs, labels)
            
            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + l2_reg * reg_loss

            # Backward propagation to compute gradient
            cost.backward()
            
            # Update parameters using gradient
            optimizer.step()

            if global_step % log_iter == 0 and writer is not None:
                writer.add_scalar('train_loss', batch_mse, global_step)

        
        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()

        mse_val /= len(val_loader.dataset)
        writer.add_scalar('val_loss', mse_val, global_step)
        print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
      
    return net
