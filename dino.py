import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers.modeling_outputs import SemanticSegmenterOutput
from seg_decoder_head import seg_decoder_head

class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class SurgicalDINO(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.

    Args:
        backbone_size: the pretrained size of dinov2 model
        r: rank of LoRA
        image_shape: input image shape
        decode_type: the decode type of decode head, "linear" or ""

    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,224), num_class=3, decode_type = 'linear4', lora_layer=None):
        super(SurgicalDINO, self).__init__()

        assert r > 0
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.image_shape = image_shape
        self.num_class = num_class
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        self.decode_type = decode_type
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.dinov2 = dinov2
        # The decode depth estimation head
        
        if self.decode_type == 'linear':
            self.inchannels = [self.embedding_dim]
            self.channels = self.embedding_dim*2
            self.in_index = (0)
            self.input_transform="resize"
        elif self.decode_type == 'linear4':
            self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim]
            self.channels = self.embedding_dim*8
            self.in_index = (0, 1, 2, 3)
            self.input_transform="resize_concat"
            
        self.decode_head = seg_decoder_head(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index,
                                              in_channels=self.inchannels, 
                                              num_class=self.num_class,
                                              channels=self.channels)
        
        # self.sig_loss = SigLoss(valid_mask=True, loss_weight=1.0, warm_up=False, loss_name='loss_depth')
        # self.gradient_loss = GradientLoss(valid_mask=True, loss_weight=0.5, loss_name='loss_grad')
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print('loaded lora parameters from %s.' % filename)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, pixel_values, depth_gt=None):
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)
        pred = self.decode_head(feature)

        loss = None
        if depth_gt is not None:
            loss = self.sig_loss(pred.squeeze(), depth_gt.squeeze()) + self.gradient_loss(pred.squeeze(), depth_gt.squeeze())
            loss = torch.mean(loss)
        return SemanticSegmenterOutput(
            loss=loss,
            predicted_depth=pred,
        )