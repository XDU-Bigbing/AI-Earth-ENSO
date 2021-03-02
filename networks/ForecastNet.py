import torch
import torch.nn as nn
import torch.nn.functional as F

class ForecastNetPlus(nn.Module):

    \\TODO 

    def __init__(self, encoder, regressor, decoder, sliding_window_size, output_seq_length, device):
        """
        Constructor
        :param encoder: (B,C,T,W,H) --> (B,D), D is the embedding of a sliding time window
        :param regressor: (B,D) --> (B,1), A sliding time window to predict the target value at a time step in the future
        :param decoder: (B,D) --> (B,C,1,W,H), A sliding time window to predict the input at a time step in the future
        :param sliding_window_size: Length of the sliding time window
        :param output_seq_length: Length of the output sequence
        """
        super(ForecastNetPlus, self).__init__()

        self.encoder = encoder
        self.regressor = regressor
        self.decoder = decoder
        self.sliding_window_size = sliding_window_size
        self.output_seq_length = output_seq_length

    def forward(self, input_seq, is_training=False):
        """
        Forward propagation of the dense ForecastNet model
        :param input_seq: Input data in the form [B,C,T,W,H]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [B,C,T,W,H] and [B,T]
        """
        # Initialise outputs
        outputs_x = torch.zeros((input_seq.shape[0], input_seq.shape[1], self.output_seq_length, input_seq.shape[3], input_seq.shape[4])).to(self.device)
        outputs_y = torch.zeros((input_seq.shape[0], self.output_seq_length)).to(self.device)

        input_seq_length = input_seq.shape[2]

        # First input
        next_cell_input = input_seq[:,:,0:self.sliding_window_size,:,:]
        for i in range(self.out_seq_length):
            hidden = self.encoder(next_cell_input)
            next_y = self.regressor(hidden)
            outputs_y[:,i] = next_y

            next_x = self.decoder(hidden)
            outputs_x[:,:,i,:,:] = next_x

            # Prepare the next input
            if is_training:
                next_cell_input = input_seq[:,:,i+1:i+self.sliding_window_size+1,:,:]
            else:
                if i+1 < input_seq_length:
                    next_cell_input = torch.cat((input_seq[:,:,i+1:input_seq_length,:,:], outputs_x[:,:,:i+1,:,:]), dim=2)
                else:
                    next_cell_input = outputs_x[:,:,i+1-self.sliding_window_size:i+1:,:]
        return outputs_x, outputs_y