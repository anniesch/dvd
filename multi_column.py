import torch.nn as nn
import torch as th
from torch.autograd import Function
import torchvision.models as models
import numpy as np
      
        
class SimilarityDiscriminator(nn.Module):
    def __init__(self, args):
        super(SimilarityDiscriminator, self).__init__()
        self.sim_layer = th.nn.Sequential(
                            nn.Linear(2*args.hidden_size, args.hidden_size),
                            nn.ReLU(),
                            nn.Linear(args.hidden_size, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                            nn.Linear(32, 2),
                            )
        self.params = list(self.sim_layer.parameters())
        
    def forward(self, enc1, enc2, alpha=1):
        # inputs have shape (b_sz, 512)
        if np.random.uniform(0.0, 1.0) < 0.5:
            enc = th.cat((enc2, enc1), axis=1)
        else:
            enc = th.cat((enc1, enc2), axis=1)
        out = self.sim_layer(enc)
        return out        


class MultiColumn(nn.Module):

    def __init__(self, args, num_classes, conv_column, column_units,
                 clf_layers=None):
        """
        - Example video encoder

        Args:
        - Input: Takes in a list of tensors each of size
                 (batch_size, 3, sequence_length, W, H)
        - Returns: features of size (batch size, column_units)
        """
        super(MultiColumn, self).__init__()
        self.column_units = column_units
        self.conv_column = conv_column(column_units)
        self.similarity = args.similarity
            
    def encode(self, inputs):
        outputs = []
        num_cols = len(inputs)
        for idx in range(num_cols):
            x = inputs[idx]
            x1 = self.conv_column(x)
            outputs.append(x1)
        outputs = th.stack(outputs).permute(1, 0, 2)
        outputs = th.squeeze(th.sum(outputs, 1), 1)
        avg_output = outputs / float(num_cols)
        return avg_output



if __name__ == "__main__":
    from model3D_1 import Model
    num_classes = 174
    input_tensor = [th.autograd.Variable(th.rand(1, 3, 72, 84, 84))]
    model = MultiColumn(174, Model, 512)
    output = model(input_tensor)
    print(output.size())
