import torch 
import torch.nn as nn
import torch.optim as optim

def train(
    net: nn.Module, 
    train_loader: torch.utils.data.DataLoader,  
    learning_rate: float = 1e-3,  
    momentum: float =0.9,
    n_epochs: int = 10,
    grad_clip: float = 10,
    log_freq: int = 10,
    device = None
):
    if device is None:
        device =  "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.MSELoss()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    loss_history = []
    i = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for _, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizes
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # if torch.isnan(loss):
            #     print("For input ", inputs)
            #     print("Output is ", outputs)
            #     print("Loss is ", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            i += 1
            if i % log_freq == log_freq - 1:    # print every 10 mini-batches
                loss_history.append(running_loss)
                running_loss = 0.0

    return loss_history