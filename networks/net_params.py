

# Encoder

## 3DCNN-Resnet


## Dilated Causal CNN
dccnn_params = {
    "in_channels": 9,
    "channels": 40,
    "depth": 7,
    "kernel_size": 3,
    "out_channels": 320,
    "reduced_size": 160
}

# Decoder

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