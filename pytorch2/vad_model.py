import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFLayer(nn.Module):
    """
    Feed-forward layer with layer-norm, dropout, activation
    """
    def __init__(self, input_size, output_size, activation):
        super(FFLayer, self).__init__()

        self.layer = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(input_size)
        
        if activation == 'relu':
            self.activation = lambda x: F.relu(x)
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise NotImplementedError

    def forward(self, inputs):
        normed = self.layer_norm(inputs)
        output = self.layer(normed)
        return self.activation(output)

    def freeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, spect_size, pcm_size, activation):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        print('latentsize:', self.latent_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # nn layers
        sizes = [self.latent_size] + self.hidden_sizes
        self.hiddens = []
        for i, s in enumerate(sizes[:-2]):
            self.hiddens.append(FFLayer(s, sizes[i+1], self.activation))
        # last layer has no activation
        self.hiddens.append(FFLayer(sizes[-1], spect_size, 'linear'))
        self.backprop_pcm = False
        if pcm_size != None:
            self.backprop_pcm = True
            self.hiddens.append(FFLayer(sizes[-1], pcm_size, 'linear')) 
        self.hiddens = nn.ModuleList(self.hiddens)

    def init_latents(self, n_points, input_d, device, mode):
        if mode == 'zeros':
            return torch.zeros(n_points, self.latent_size, dtype=torch.float,
                        requires_grad=True, device=device)
        elif mode == 'normal':
            stdev = 0.001
            print("Initializing from normal distribution with stdev {}".format(stdev), file=sys.stderr)

            return torch.tensor(np.random.normal(0, stdev, size=(n_points, input_d, self.latent_size)), dtype=torch.float,
                requires_grad=True, device=device)

        else:
            raise NotImplementedError

    def forward(self, latents):
        x = latents
        results = []
        if self.backprop_pcm:
            for layer in self.hiddens[:-2]:
                x = layer(x)
            spect = self.hiddens[-2](x)
            pcm = self.hiddens[-1](x)
            results.append(spect)
            results.append(pcm)
        else:
            for layer in self.hiddens:
                x = layer(x)
            results.append(x)
        mask = torch.ones_like(results[0])
        return results, mask

    def freeze_hiddens(self):
        for h in self.hiddens:
            h.freeze_weights()

    def unfreeze_hiddens(self):
        for h in self.hiddens:
            h.unfreeze_weights()


class LSTM(nn.Module):

    def __init__(self, latent_size, hidden_layers, output_size, layer_norm, dropout, activation):
        super(LSTM, self).__init__()

        self.latent_size = latent_size
        print('latentsize:', self.latent_size)
        
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # lstm layer
        self.lstm = nn.LSTM(self.latent_size, self.latent_size, self.hidden_layers, batch_first = True)
        self.linear = nn.Linear(self.latent_size, self.output_size)


    def init_latents(self, n_points, input_d, device, mode):
        if mode == 'zeros':
            return torch.zeros(n_points, self.latent_size, dtype=torch.float,
                        requires_grad=True, device=device)
        elif mode == 'normal':
            stdev = 0.001
            print("Initializing from normal distribution with stdev {}".format(stdev), file=sys.stderr)

            return torch.tensor(np.random.normal(0, stdev, size=(n_points, input_d, self.latent_size)), dtype=torch.float,
                requires_grad=True, device=device)

            #return torch.randn(n_points, self.latent_size, dtype=torch.float,
            #            requires_grad=True, device=device) * stdev
        else:
            raise NotImplementedError

    def forward(self, latents):
        # latents = latents.transpose(0,1)
        # print('latent size is now', latents.shape)

        lstm_out, self.hidden = self.lstm(latents)
        y_pred = self.linear(lstm_out)
        # print('y_pred size is now', y_pred.shape)
        return y_pred
    def freeze_hiddens(self):
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

    def unfreeze_hiddens(self):
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = True

