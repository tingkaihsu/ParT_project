import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }
    
    # For regression
    # def forward(self, points, features, lorentz_vectors, mask):
    #     return self.mod(features, v=lorentz_vectors, mask=mask)
    
    # For multi-class classification
    # def forward(self, points, features, lorentz_vectors, mask):
    #     logits = self.mod(features, v=lorentz_vectors, mask=mask)
    #     return torch.softmax(logits, dim=-1)  # Apply softmax for multi-class classification
    
    def forward(self, points, features, lorentz_vectors, mask):
        logits = self.mod(features, v=lorentz_vectors, mask=mask)
        return logits  # Return raw logits without softmax

def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # activation = 'relu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},

        # 'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        # 'output_names': ['linear'],
        # 'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'linear': {0: 'N'}}},
    }
    return model, model_info

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    # return torch.nn.MSELoss()   # MSE for regression