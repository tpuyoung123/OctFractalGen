"""Build and load the vendored OctGPT VQVAE."""
import torch

from models.vae import VQVAE


def build_vqvae(ckpt_path, device='cuda', freeze=True):
    """Build and load the pretrained large VQVAE (BSQ32).

    Config matches octgpt/configs/ShapeNet/shapenet_vae.yaml:
      name: vqvae_large, in_channels: 4, embedding_channels: 32,
      quantizer_type: bsq, feature: ND
    """
    class VQVAEl(VQVAE):
        """Large VQVAE variant (33.9M params), matching octgpt/utils/builder.py."""

        def config_network(self):
            self.bottleneck = 1
            self.mpu_stage_nums = 3
            self.pred_stage_nums = 3

            self.enc_channels = [32, 32, 64]
            self.enc_resblk_nums = [2, 2, 2]

            self.dec_enc_channels = [64, 128, 256, 512]
            self.dec_enc_resblk_nums = [2, 4, 8, 2]
            self.dec_dec_channels = [512, 256, 128, 64, 32, 32]
            self.dec_dec_resblk_nums = [2, 4, 8, 2, 2, 2]

    vqvae = VQVAEl(
        in_channels=4,
        embedding_sizes=128,
        embedding_channels=32,
        feature='ND',
        n_node_type=7,
        quantizer_type='bsq',
        quantizer_group=4,
        rnd_flip=0.0,
    )
    ck = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    vqvae.load_state_dict(ck)
    vqvae.to(device)
    if freeze:
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad = False
    return vqvae
