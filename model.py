import log
logger = logging.getLogger('root') 

import torch
import torch.nn as nn


class FlattenLinearProjection(nn.Module):
    def __init__(self, input_dim, out_dim, hl_scalings=[]):
        super().__init__()
        if hl_scalings:
            hl_dims = self._process_hl_scalings(input_dim, hl_scalings)
        self.linears = nn.Sequential(
            self._process_hl_dims(input_dim, hl_dims, out_dim)
        )

    def _process_hl_scalings(self, input_dim, hl_scalings):
        """
            note that this function truncates when numbers results in float. (e.g. 5*1.5 = 7)
        """
        layers_dims = []
        prev = curr = input_dim
        for curr in hl_scalings:
            if curr == None:
                msg = f"'None' value encountered for hidden_layers. Assuming same dim as previous layer is desired={prev}."
                logger.warning(msg)
                curr = prev
            curr = int(layers_dims*curr)
            layers_dims.append(curr)
            prev = curr
        return layers_dims

    def _process_hl_dims(self, input_dim, hl_dim, out_dim):
        linear_layers = []
        prev = curr = input_dim # = curr is needed if len(hl_dim) == 0
        for curr in hl_dim:
            linear_layers.append(
                nn.Linear(prev, curr)
            )
            prev = curr
        linear_layers.append(
            nn.Linear(curr, out_dim)
        )
        return linear_layers
            
    def forward(self, batch):
        return self.linears(batch.flatten(start_dim=-2)) # Batch x Tiles x Height x Width


class ThoracicAbnormalityModel(nn.Module):
    def __init__(self, ds, linear_projection_configs, encoder_configs, num_residual_blocks, rpn_configs):
        super().__init__()
        linear_projection_configs = self._preprocess_linear_projection_configs(ds, linear_projection_configs)
        self.flp = FlattenLinearProjection(**linear_projection_configs)
        # TODO 1) position embeddings appended onto projection
        #      2) Transformer Encoder
        #      3) Residual Blocks
        #      4) RPN
        #      5) output

    def _preprocess_linear_projection_configs(self, ds, lp_configs):
        tile_x_dim, tile_y_dim = ds[0].shape[-2:]
        flat_dim = tile_x_dim * tile_y_dim
        if not lp_configs.get('input_dim'):
            lp_configs['input_dim'] = flat_dim
        if not lp_configs.get('output_dim'):
            lp_configs['output_dim'] = flat_dim 
        return lp_configs
    
    def forward(self, X):
        self.flp(X)
