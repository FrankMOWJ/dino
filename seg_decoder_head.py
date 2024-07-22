import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def compute_segmentation_loss(pred, target, ignore_index=None):
    """
    计算n个类别的语义分割交叉熵损失。
    
    Args:
        pred (torch.Tensor): 模型的预测输出，形状为 (N, num_class, H, W)
        target (torch.Tensor): 实际标签，形状为 (N, H, W)
        ignore_index (int, optional): 需要忽略的标签索引。默认值为 None。
    
    Returns:
        torch.Tensor: 计算的交叉熵损失。
    """
    # 如果需要忽略某些标签
    if ignore_index is not None:
        loss = F.cross_entropy(pred, target, ignore_index=ignore_index)
    else:
        loss = F.cross_entropy(pred, target)
    
    return loss

class seg_decoder_head(torch.nn.Module):
    def __init__(self, input_transform="resize_concat", image_shape=(224,224), in_index=(0, 1, 2, 3), upsample=4, num_class=3,
       in_channels=[768,768,768,768], channels=6144, align_corners=False, scale_up=False):
        super().__init__()
        self.input_transform = input_transform
        self.image_shape = image_shape
        self.in_index = in_index
        self.upsample = upsample
        self.num_classes = num_class
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.scale_up = scale_up
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.bn = nn.SyncBatchNorm(self.in_channels)

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1, padding=0, stride=1)
        
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, img_metas=None, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        # surgical dino 没有用， 但是dinov2 decoder用来了
        feats = self.bn(x)
        return feats
    
    def seg_pred(self, feat):
        """Prediction each pixel."""
        logit = self.conv_seg(feat) # (bs, 1, w, h)

        # output = torch.softmax(logit, dim=1) 

        return logit
    
    def forward(self, inputs, **kwargs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.seg_pred(output)
        output = torch.nn.functional.interpolate(output, size=self.image_shape, mode="bilinear", align_corners=self.align_corners)

        return output
    

if __name__ == "__main__":
    decoder = seg_decoder_head()
    criterion = compute_segmentation_loss
    
    multi_level_feat = [torch.rand(1, 16, 224, 224), torch.rand(1, 32, 112, 112), \
                        torch.rand(1, 64, 112, 112), torch.rand(1,64, 54, 54)]
    

    decoder_output = decoder(multi_level_feat)
    print(decoder_output.shape)

