import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
        
        # # Event feature processing
        # self.event_fc = torch.nn.Sequential(
        #     torch.nn.Linear(3, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 128)
        # )
        
        # Modified classifier - input size will be determined dynamically
        self.classifier = torch.nn.Linear(256, kwargs['num_classes'])

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }
    
    # def forward(self, points, features, lorentz_vectors, mask, event_features):
    #     # Process particles
    #     particle_output = self.mod(features, v=lorentz_vectors, mask=mask)
        
    #     # Get transformer output dimension
    #     transformer_dim = particle_output.size(-1)
        
    #     # Create dynamic projection layer matching transformer output
    #     projection = torch.nn.Linear(transformer_dim, 128).to(particle_output.device, particle_output.dtype)
        
    #     # Apply projection
    #     particle_output = projection(particle_output)
        
    #     # Process event features
    #     event_features = event_features.squeeze(-1)
    #     event_embedding = self.event_fc(event_features)
        
    #     # Combine features
    #     combined = torch.cat([particle_output, event_embedding], dim=1)
    #     logits = self.classifier(combined)
    #     return logits
    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
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
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            **{'softmax': {0: 'N'}}
        }}
    return model, model_info

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()