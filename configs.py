DATA = {
    "summary_csv": "summary.csv",
    "root_dir": "data/",
    "pre_split": True
}

MODEL = {
    "linear_projection_configs": {
        "hl_dim_scalings":[]
    },
    "encoder_configs": {
        "num_layers": 3,
        "nhead": 3,
        "dim_ff":2048,
        "dropout":0.1,
        "activation":"relu"
    },
    "res_block_configs": {
        "block_channels": [1,1,1,1] # e.g. [(3,6,3),(3,5,3)]
    },
    "rpn_configs" : {}
}