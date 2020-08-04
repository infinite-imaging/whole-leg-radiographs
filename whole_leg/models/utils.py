import torch


class ConvWrapper(torch.nn.Module):
    def __init__(
        self,
        n_dim,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transposed=False,
        **kwargs
    ):

        super().__init__()

        if transposed:
            transposed_str = "Transpose"
        else:
            transposed_str = ""

        conv_cls = getattr(torch.nn, "Conv%s%dd" % (transposed_str, n_dim))

        self.conv = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )

    def forward(self, x: torch.Tensor):

        return self.conv(x)


class PoolingWrapper(torch.nn.Module):
    def __init__(self, pooling_type, n_dim, *args, **kwargs):

        super().__init__()

        pool_cls = getattr(torch.nn, "%sPool%dd" % (pooling_type, n_dim))

        self.pool = pool_cls(*args, **kwargs)

    def forward(self, x: torch.Tensor):

        return self.pool(x)


class NormWrapper(torch.nn.Module):
    def __init__(self, norm_type, n_dim, *args, **kwargs):

        super().__init__()
        if norm_type is None:
            self.norm = None
        else:
            if n_dim is None:
                dim_str = ""
            else:
                dim_str = str(n_dim)

            norm_cls = getattr(torch.nn, "%sNorm%sd" % (norm_type, dim_str))
            self.norm = norm_cls(*args, **kwargs)

    def forward(self, x: torch.Tensor):

        if self.norm is None:
            return x
        else:
            return self.norm(x)


class DropoutWrapper(torch.nn.Module):
    def __init__(self, n_dim, p=0.5, inplace=False):
        super().__init__()
        dropout_cls = getattr(torch.nn, "Dropout%dd" % n_dim)
        self.dropout = dropout_cls(p=p, inplace=inplace)

    def forward(self, x):
        return self.dropout(x)
