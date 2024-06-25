
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, physics_kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                            out_channels=4 * self.hidden_dim,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)

        self.physics_conv_x = nn.Conv2d(in_channels=self.input_dim,
                                        out_channels=self.hidden_dim,
                                        kernel_size=physics_kernel_size,
                                        padding=physics_kernel_size[0] // 2,
                                        bias=False)

        self.physics_conv_y = nn.Conv2d(in_channels=self.input_dim,
                                        out_channels=self.hidden_dim,
                                        kernel_size=physics_kernel_size,
                                        padding=physics_kernel_size[1] // 2,
                                        bias=False)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Check if input_tensor has an extra dimension (sequence length)
        if input_tensor.dim() == 5:
            input_tensor = input_tensor.squeeze(1)

        # Ensure the number of channels in input_tensor matches the input_dim
        if input_tensor.size(1) != self.input_dim:
            raise ValueError(f"Expected input_tensor to have {self.input_dim} channels, but got {input_tensor.size(1)} channels instead")

        # Ensure the number of channels in h_cur matches the hidden_dim
        if h_cur.size(1) != self.hidden_dim:
            raise ValueError(f"Expected h_cur to have {self.hidden_dim} channels, but got {h_cur.size(1)} channels instead")

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Compute physics-based convolutions
        physics_conv_x = self.physics_conv_x(input_tensor)
        physics_conv_y = self.physics_conv_y(input_tensor)

        i = torch.sigmoid(cc_i + physics_conv_x)
        f = torch.sigmoid(cc_f + physics_conv_x)
        o = torch.sigmoid(cc_o + physics_conv_y)
        g = torch.tanh(cc_g + physics_conv_y)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, physics_kernel_size,output_dim,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          physics_kernel_size=physics_kernel_size))

        self.cell_list = nn.ModuleList(cell_list)
        self.output_conv = nn.Conv2d(in_channels=hidden_dim[-1],
                                      out_channels=output_dim,
                                      kernel_size=1,
                                      padding=0)

    def forward(self, input_tensor, hidden_state=None):
        if input_tensor.dim() == 4:
            # (b, h, w, c) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(0, 3, 1, 2).unsqueeze(1)
        elif input_tensor.dim() == 5:
            if not self.batch_first:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # Remove the sequence length dimension before applying the output convolution
        output = self.output_conv(layer_output_list[0].squeeze(1))
        # Permute the output to have shape (b, h, w, c)
        output = output.permute(0, 2, 3, 1)
        return output, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def reset_parameters(self):
        """
        Reset all parameters of the model.
        """
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LSTM, nn.LSTMCell)):
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                if module.affine:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)

        # If the model has any custom parameters, reset them here
        # For example:
        # if hasattr(self, 'custom_weight'):
        #     nn.init.xavier_uniform_(self.custom_weight)
        # if hasattr(self, 'custom_bias'):
        #     nn.init.zeros_(self.custom_bias)

        print(f"Parameters of {self.__class__.__name__} have been reset.")