import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(
            x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value
        )
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


def LightweightConv(
    input_size,
    kernel_size=1,
    padding_l=None,
    num_heads=1,
    weight_dropout=0.0,
    weight_softmax=False,
    bias=False,
):
    return LightweightConv1dTBC(
        input_size,
        kernel_size=kernel_size,
        padding_l=padding_l,
        num_heads=num_heads,
        weight_dropout=weight_dropout,
        weight_softmax=weight_softmax,
        bias=bias,
    )


class LightweightConv1d(nn.Module):
    """Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding=0,
        num_heads=1,
        weight_softmax=False,
        bias=False,
        weight_dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zero(input_size))
        else:
            self.bias = None
        self.weight_dropout_module = nn.Dropout(weight_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = torch.softmax(weight, dim=-1)

        weight = self.weight_dropout_module(weight)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.reshape(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.reshape(B, C, T)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1)

        return output


class LightweightConv1dTBC(nn.Module):
    """Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding_l=None,
        num_heads=1,
        weight_dropout=0.0,
        weight_softmax=False,
        bias=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout_module = nn.Dropout(weight_dropout)
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(input_size))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, unfold=False):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """

        if unfold:
            output = self._forward_unfolded(x)
        else:
            output = self._forward_expanded(x)

        if self.bias is not None:
            output = output + self.bias.reshape(1, 1, -1)
        return output

    def _forward_unfolded(self, x):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.reshape(H, K)

        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
        x_unfold = x_unfold.reshape(T * B * H, R, K)

        if self.weight_softmax:
            weight = torch.softmax(weight, dim=1)

        weight = (
            weight.reshape(1, H, K)
            .expand(T * B, H, K)
            .contiguous()
            .reshape(T * B * H, K, 1)
        )

        weight = self.weight_dropout_module(weight)
        output = torch.bmm(x_unfold, weight)  # T*B*H x R x 1
        output = output.reshape(T, B, C)
        return output

    def _forward_expanded(self, x):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.reshape(H, K)
        if self.weight_softmax:
            weight = torch.softmax(weight, dim=1)

        weight = weight.reshape(1, H, K).expand(T * B, H, K).contiguous()
        weight = weight.reshape(T, B * H, K).transpose(0, 1)

        x = x.reshape(T, B * H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
        weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(
            weight
        )
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = self.weight_dropout_module(weight_expanded)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().reshape(T, B, C)
        return output

    def extra_repr(self):
        s = "{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}".format(
            self.input_size,
            self.kernel_size,
            self.padding_l,
            self.num_heads,
            self.weight_softmax,
            self.bias is not None,
        )
        if self.weight_dropout_module.p > 0.0:
            s += ", weight_dropout={}".format(self.weight_dropout_module.p)
        return s


class LightConvEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, config, kernel_size=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.conv_dim = config.hidden_size
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        if config.use_glu:
            self.Linear1 = nn.Linear(self.hidden_size, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.Linear1 = nn.Linear(self.hidden_size, self.conv_dim)
            self.act = None

        self.conv = LightweightConv(
            self.conv_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=config.weight_softmax,
            num_heads=config.num_attention_heads,
            weight_dropout=config.attention_probs_dropout_prob,
        )

        self.Linear2 = nn.Linear(self.conv_dim, self.hidden_size)

        self.dropout_module = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.act_dropout_module = nn.Dropout(config.hidden_dropout_prob)
        self.input_dropout_module = nn.Dropout(config.hidden_dropout_prob)

        self.fc1 = nn.Linear(self.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.hidden_size)

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size, config.layer_norm_eps) for _ in range(2)]
        )

    def forward(self, x, attention_mask=None, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch,s eq_len, embed_dim)`
            attention_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``0``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.input_dropout_module(x)
        x = self.Linear1(x)
        if self.act is not None:
            x = self.act(x)
        if attention_mask is not None:
            x = x.masked_fill(attention_mask[:, :, None] == 0, 0)
        x = self.conv(x.transpose(0, 1)).transpose(0, 1)
        x = self.Linear2(x)
        x = self.dropout_module(x)
        x = self.layer_norms[0](residual + x)

        residual = x
        x = self.intermediate_act_fn(self.fc1(x))
        x = self.act_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.layer_norms[1](residual + x)
        return (x, None) if output_attentions else (x,)
