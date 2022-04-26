import pandas as pd
import torch
import json
import ast


class LSTM(torch.nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):

        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
#         print(self.lstm.device)
#         print(self.linear.device)
        
    def forward(self, x, rank, hidden = None):

        if hidden == None:

            self.hidden = (torch.zeros(1,1,self.hidden_size).to(rank),
                           torch.zeros(1,1,self.hidden_size).to(rank))

        else:

            self.hidden = hidden
  
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """

        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        
        predictions = self.linear(lstm_out.view(len(x), -1))
        
        return predictions[-1]
    
    
class LSTM_CPU(torch.nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size):

        super(LSTM_CPU, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
#         print(self.lstm.device)
#         print(self.linear.device)
        
    def forward(self, x, hidden = None):

        if hidden == None:

            self.hidden = (torch.zeros(1,1,self.hidden_size),
                           torch.zeros(1,1,self.hidden_size))

        else:

            self.hidden = hidden
  
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """

        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        
        predictions = self.linear(lstm_out.view(len(x), -1))
        
        return predictions[-1]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":

    model = LSTM(input_size = 512,
                 hidden_size = 128,
                 output_size = 12)    

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)                 

    with open("/sciclone/geograd/heather_data/temporal_features/jsons/484005031.json", "r") as f:
        data = json.load(f)
        
    features = list(data.values())
    migs = [torch.tensor([ast.literal_eval(i["migrants"])]) for i in features]
    features = [torch.tensor([ast.literal_eval(i["features: "])]) for i in features]

    num_steps = 12
    x, y = [], []
    for i in range(len(features)):
        x.append(torch.cat(features[i:i+num_steps]))
        y.append(torch.cat(migs[i:i+num_steps]))


    x = torch.cat([i.unsqueeze(0) for i in x if i.shape[0] == 12])
    # y = [i for i in y if i.shape[0] == 12]

    train_num = int(x.shape[0] * .75)

    print("Train num: ", train_num)

    import random

    train_indices = random.sample(range(x.shape[0]), train_num)
    val_indices = [i for i in range(x.shape[0]) if i not in train_indices]

    x_train = torch.index_select(x, 0, torch.tensor(train_indices))
    x_val = torch.index_select(x, 0, torch.tensor(val_indices))

    print(x_train.shape, x_val.shape)

    # train_tracker = AverageMeter()

    # for epoch in range(0, 100):

    #     train_tracker.reset()

    #     for input, target in zip(x_train, y_train):

    #         if input.shape[0] == 12:
            
    #             output = model(input)
    #             loss = criterion(target, output)

    #             train_tracker.update(loss.item())

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #     print(train_tracker.avg)
