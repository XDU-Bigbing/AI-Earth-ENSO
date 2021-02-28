import torch.nn as nn
import torch


# 一个神经单元
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, padding, dilation):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            通道数
        hidden_dim: int
            隐藏层通道数
        kernel_size: (int, int)
            卷积核大小
        bias: bool
            是否使用 bias
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # 每个维度填充点数
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              dilation=self.dilation,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # 上一时刻的输出 
        h_cur, c_cur = cur_state

        # dim=1 横向拼接
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        # 按照 self.hidden_dim 切分数据 dim=1 横向拆分
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 记忆门
        i = torch.sigmoid(cc_i)
        # 遗忘门
        f = torch.sigmoid(cc_f)
        # 输出门
        o = torch.sigmoid(cc_o)
        # 当前状态
        g = torch.tanh(cc_g)

        # 更新状态
        # shape : [batchsize, hidden_dim, height, width]
        c_next = f * c_cur + i * g
        # 产生输出
        h_next = o * torch.tanh(c_next)

        # 返回输出与状态
        return h_next, c_next

    # 用 0 初始化参数
    def init_hidden(self, batch_size, length, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, length, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, length, height, width, device=self.conv.weight.device))


# 神经单元组成的神经网络
class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        T 是序列长度
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 padding, dilation, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # 检查类型是否为元组或列表
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # 里面元素复制三份
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # 列表类型不处理
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            # h 的输出维度是 hidden_dim，所以能对接上
            # [channel, hidden_dim] [hidden_dim, hidden_dim]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding = self.padding,
                                          dilation = self.dilation,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Returns
        -------
        last_state_list, layer_output
        """

        layer_output_list = []
        last_state_list = []

        # 序列长度
        length = input_tensor.size(2)

        # batch, height, width
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # 初试化状态为 0
            hidden_state = self._init_hidden(batch_size=b,
                                             length=length,
                                             image_size=(h, w))

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            # 读入一个序列
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, :, :, :, :],
                                             cur_state=[h, c])

            # 每层网络之间的状态 c 是断的
            # 一个序列在一层的输出
            # 下一层的输入
            # cur_layer_input = output_inner[-1]
            cur_layer_input = h

            layer_output_list.append(h)
            last_state_list.append([h, c])

        # 如果只要最后一层的输出，否则返回每一层的输出
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    # 初始化这一层的参数
    def _init_hidden(self, batch_size, length, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, length, image_size))
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


if __name__ == "__main__":

    in_channels = 36
    out_channels = 16
    length = 12
    num_layers = 3
    hidden_dim = [32, 64, 128]
    kernel_size = (3, 3, 3)
    height, width = 24, 72
    dilation = (2, 2, 2)
    batch_size = 8

    padding = tuple(dilation[i] * (kernel_size[i] - 1) // 2 for i in range(3))

    # (batch大小, 通道数, 序列长度, 高度, 宽度)
    x = torch.rand((batch_size, in_channels, length, height, width))

    # 输入维度, 隐藏维度, kernel大小, 层数，batch在先，含有偏执项，返回全部层
    convlstm = ConvLSTM(input_dim=in_channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    num_layers=num_layers,
                    bias=True,
                    return_all_layers=False)
    
    print(convlstm)

    # 返回最后的状态
    _, last_states = convlstm(x)

    # 第一个 0 是层的 index，第二个 0 是 h 状态的索引
    h = last_states[0][0]

    print(x.size())
    print(h.size())

    '''
    ConvLSTM(
        (cell_list): ModuleList(
            (0): ConvLSTMCell(    
            (conv): Conv3d(68, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2))
            )
            (1): ConvLSTMCell(
            )
            (2): ConvLSTMCell(
            (conv): Conv3d(192, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2))
            )
        )
    )
    in : torch.Size([8, 36, 12, 24, 72])
    out : torch.Size([8, 128, 12, 24, 72])
    '''