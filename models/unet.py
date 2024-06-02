import torch
import torch.nn as nn

from .model_utils import Conv2dBlock, LearnedTimeEmbedding, Activation, ResNetBlock, Attention, Downsample, Upsample

class UNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        self.model_params = model_params

        self.input_channels = self.model_params["input_channels"]
        self.ouptut_channels = self.model_params["output_channels"]
        self.initial_channels = self.model_params["initial_channels"]
        self.resolution_channels = self.model_params["resolution_channels"]
        self.norm_type = self.model_params["norm_type"]
        self.norm_groups = self.model_params["norm_groups"]
        self.activation = self.model_params["activation"]
        self.attention_resolution = self.model_params["attention_resolution"]
        self.resblock_count = self.model_params["resblock_count"]
        self.time_embedding_dim = self.model_params["time_embedding_dim"]
        self.attention_heads = self.model_params["attention_heads"]
        self.middle_unet_blocks = self.model_params["middle_unet_blocks"]

        self.num_classes = self.model_params["num_classes"]

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, self.time_embedding_dim)

        self.initial_conv = Conv2dBlock(
            in_channels=self.input_channels, out_channels=self.initial_channels, norm_type=self.norm_type, activation=self.activation, norm_groups=self.norm_groups, kernel_size=3, padding=1
        )

        self.time_mlp = nn.Sequential(
            LearnedTimeEmbedding(self.time_embedding_dim),
            nn.Linear(self.time_embedding_dim + 1, self.time_embedding_dim),
            Activation(activation=self.activation),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        )

        down_layers = []

        current_channels = self.initial_channels

        for i, (channels, r_count, has_attn) in enumerate(zip(self.resolution_channels, self.resblock_count, self.attention_resolution)):
            for _ in range(r_count):
                down_layers.append(
                    ResNetBlock(in_channels=current_channels, out_channels=channels, time_emb_dim=self.time_embedding_dim, norm_type=self.norm_type, norm_groups=self.norm_groups, activation=self.activation)
                )

                current_channels = channels

            if has_attn:
                down_layers.append(
                    Attention(channels=current_channels, num_heads=self.attention_heads, skip_connect=True, activation=self.activation, norm_type=self.norm_type, norm_groups=self.norm_groups)
                )
            
            if i != len(self.resolution_channels) - 1:
                down_layers.append(
                    Downsample(input_channels=current_channels, output_channels=self.resolution_channels[i+1], factor=2)
                )

                current_channels = self.resolution_channels[i+1]

        self.down_layers = nn.ModuleList(down_layers)      

        mid_layers = []
        for layer in self.middle_unet_blocks:
            if layer == "res":
                mid_layers.append(
                    ResNetBlock(in_channels=current_channels, out_channels=current_channels, time_emb_dim=self.time_embedding_dim, norm_type=self.norm_type, norm_groups=self.norm_groups, activation=self.activation)
                )
            elif layer == "attn":
                mid_layers.append(
                    Attention(channels=current_channels, num_heads=self.attention_heads, skip_connect=True, activation=self.activation, norm_type=self.norm_type, norm_groups=self.norm_groups)
                )
            else:
                NotImplementedError(f"[UNet] currently not implement: {layer}")
        
        self.mid_layers = nn.ModuleList(mid_layers)

        up_layers = []
        for i, (channels, r_count, has_attn) in enumerate(zip(reversed(self.resolution_channels), reversed(self.resblock_count), reversed(self.attention_resolution))):
            output_channels = self.initial_channels
            if i != len(self.resolution_channels) - 1:
                output_channels = list(reversed(self.resolution_channels))[i+1]
                up_layers.append(
                    Upsample(input_channels=current_channels, output_channels=output_channels, factor=2)
                )

                # skip connection.
                current_channels = 2 * output_channels

            for i in range(r_count):
                if i == 0:
                    up_layers.append(
                        ResNetBlock(
                            in_channels=current_channels, out_channels=output_channels, time_emb_dim=self.time_embedding_dim, norm_type=self.norm_type, norm_groups=self.norm_groups, activation=self.activation
                        )
                    )
                    current_channels = output_channels
                else:
                    up_layers.append(
                        ResNetBlock(
                            in_channels=current_channels, out_channels=current_channels, time_emb_dim=self.time_embedding_dim, norm_type=self.norm_type, norm_groups=self.norm_groups, activation=self.activation
                        )
                    )
                
            if has_attn:
                up_layers.append(
                    Attention(channels=current_channels, num_heads=self.attention_heads, skip_connect=True, activation=self.activation, norm_type=self.norm_type, norm_groups=self.norm_groups)
                )

        self.up_layers = nn.ModuleList(up_layers)

        self.final_res_block = ResNetBlock(in_channels=current_channels + self.initial_channels, out_channels=current_channels, time_emb_dim=self.time_embedding_dim, norm_type=self.norm_type, norm_groups=self.norm_groups, activation=self.activation)
        self.final_conv = Conv2dBlock(in_channels=current_channels, out_channels=self.ouptut_channels, kernel_size=1, padding=0, activation='none')

    def forward(self, x, time, y=None):
        assert (y is not None) == (self.num_classes is not None), "must specify the number of classes if the model is class-conditioned"

        x = self.initial_conv(x)
        skip = x.clone()

        time = self.time_mlp(time)
        if self.num_classes is not None:
            time = time + self.label_emb(y)
        
        unet_skips = []

        for layer in self.down_layers:
            if isinstance(layer, ResNetBlock):
                x = layer(x, time)
            elif isinstance(layer, Downsample):
                unet_skips.append(x)
                x = layer(x)
            else:
                x = layer(x)

        for layer in self.mid_layers:
            if isinstance(layer, ResNetBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        for layer in self.up_layers:
            if isinstance(layer, ResNetBlock):
                x = layer(x, time)
            elif isinstance(layer, Upsample):
                x = layer(x)
                x = torch.cat((x, unet_skips.pop()), dim=1)
            else:
                x = layer(x)

        x = torch.cat((x, skip), dim=1)
        x = self.final_res_block(x, time)

        return self.final_conv(x)  
