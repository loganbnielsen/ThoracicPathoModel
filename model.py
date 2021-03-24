import log
import logging
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


class ThoracicPathoModel(nn.Module):
    def __init__(self, ds, linear_projection_configs, encoder_configs, res_block_configs, rpn_configs):
        super().__init__()
        # projection
        linear_projection_configs = self._preprocess_linear_projection_configs(ds, linear_projection_configs)
        self.flp = FlattenLinearProjection(**linear_projection_configs)
        # embedding
        encoder_configs = self._preprocess_embedding_configs(ds)
        self.embeddings = nn.Embedding(**encoder_configs)
        # transformer encoder
        encoder_configs = self._preprocess_encoder_configs(ds, encoder_configs)
        self.encoder = TransformerEncoderWrapper(**encoder_configs)
        # residual blocks
        self.res_blocks = ResidualBlocks(**res_block_configs)
        # TODO I'm here.
        #      4) RPN
        #      5) output


        self.num_tiles = ds.num_splits
        self.num_x_splits = ds.num_x_splits
        self.num_y_splits = ds.num_y_splits

    def _preprocess_linear_projection_configs(self, ds, lp_configs):
        tile_x_dim, tile_y_dim = ds[0].shape[-2:]
        flat_dim = tile_x_dim * tile_y_dim
        if not lp_configs.get('input_dim'):
            lp_configs['input_dim'] = flat_dim
        if not lp_configs.get('output_dim'):
            lp_configs['output_dim'] = flat_dim 
        self.tile_x_dim, self.tile_y_dim = tile_x_dim, tile_y_dim
        return lp_configs

    def _preprocess_embedding_configs(self, ds):
        embedding_configs = dict()
        if not embedding_configs.get('num_embeddings'):
            embedding_configs['num_embeddings'] = ds.number_of_splits
        if not embedding_configs.get('embedding_dim'):
            height, width = ds.shape
            embedding_configs['embedding_dim'] = height*width
        return embedding_configs

    def _preprocess_encoder_configs(self, ds, encoder_configs):
        if not encoder_configs.get('dmodel'):
            num_splits, (x_dim, y_dim)  = ds.num_splits, ds.shape
            encoder_configs['dmodel'] = num_splits * x_dim * y_dim
        return encoder_configs

    def _combine_tiles(self, X, tile_dim=-3):
        """
            Assemble the feature tiles into cohesive feature image

            Batch x Tiles x Height x Width -> Batch x Height x Width
        """
        X = X.reshape(-1, self.num_x_splits, self.num_y_splits, self.tile_x_dim, self.tile_y_dim)
        X = X.permute(0,1,3,2,4).reshape(-1,self.tile_x_dim*self.num_x_splits,
                                            self.tile_y_dim*self.num_y_splits)
        return X


    
    def forward(self, X):
        X = self.flp(X)
        for i in range(self.num_tiles):
            X[:,i,:] += self.embeddings(i) # Batch x Tiles x Flattened
        X = self.encoder(X)
        X = self._combine_tiles(X) # "Reinterpret transformer patch outputs as a feature map"
        # Batch x Tiles x Height x Width -> Batch x Height x Width

    


class TransformerEncoderWrapper(nn.Module):
    """
        Wraps what pytorch refers to as a `TransformerEncoderLayer` with the `TransformerEncoder`
        to treat them as one module
    """
    def __init__(self, num_layers, dmodel, nhead, dim_ff=2048, dropout=0.1, activation="relu"):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nhead,
                                           dim_feedforward=dim_ff,
                                           dropout=dropout, activation=activation)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
    
    def forward(self, X):
        return self.encoder(X)


class ResidualBlocks(nn.Module):
    def __init__(self, block_channels):
        """
            [(3,6,3),(3,5,3)] TODO make it so input and output channels for a block don't need to match.
        """
        super().__init__()
        self.blocks = nn.Sequential([ResBlock(*c) for c in block_channels])
    
    def forward(self, X):
        return self.blocks(X)


class ResBlock(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__(channels)
        self.activ = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, X):
        orig_X = X

        X = self.conv1(X)
        X = self.activ(self.bn1(X))

        X = self.conv2(X)
        X = self.activ(self.bn2(X))

        return orig_X + X

        
