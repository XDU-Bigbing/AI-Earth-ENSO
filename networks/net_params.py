

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
decoder_params = [
    # Linear 1
    {
        "in_features": 320,
        "out_features": 512,
        "bias" : True
    },
    # Linear 2
    {
        "in_features": 512,
        "out_features": 1024,
        "bias" : True
    },
    # Linear 3
    {
        "in_features": 1024,
        "out_features": 4096,
        "bias" : True
    },
    # Linear 4
    {
        "in_features": 4096,
        "out_features": 6912,
        "bias" : True
    },
]

decoder_channels = 4
decoder_H = 24
decoder_W = 72

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