

# Encoder

## 3DCNN-Resnet


## Dilated Causal CNN
dccnn_params = {
    "in_channels": 512,
    "channels": 40,
    "depth": 3,
    "kernel_size": 3,
    "out_channels": 320,
    "reduced_size": 160
}

# Decoder
decoder_params = {
    "in_channels":320,
    "reduced_size":160,
    "input_dim":160,
    "upsample_size_one":(12, 36),
    "upsample_size_two":(24, 72),
    "kernel_size":3,
    "stride":1,
    "padding":1,
    "out_padding":1,
}

# Regressor
regressor_params = [
    # Linear 1
    {
        "in_features": 320,
        "out_features": 128,
        "bias" : True
    },
    # Linear 2
    {
        "in_features": 128,
        "out_features": 32,
        "bias" : True
    },
    # Linear 3
    {
        "in_features": 32,
        "out_features": 8,
        "bias" : True
    },
    # Linear 4
    {
        "in_features": 8,
        "out_features": 1,
        "bias" : True
    },
]

# ForecastNet
sliding_window_size = 12
output_seq_length = 24