#!/usr/bin/env python3
"""
Evaluate a trained MANTiS Stage 1 checkpoint on downstream classification.

This script runs the following pipeline:
Image -> [Stem -> Encoder] -> VIB -> Decoder -> [Pretrained Tail] -> Logits

It measures two key metrics:
1.  Downstream Accuracy: Top-1 and Top-5 accuracy on ImageNet-1k.
2.  Rate (BPP): Bits per pixel required to transmit the latent tensor 'z'.
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Assuming the training script's models are available or redefined here
from client.models import MANTiSClient
from server.decoders_tails import FrankenSplitDecoder, ResNetCompatibleTail
from vib import VIBBottleneck
from webdataset_wrapper import create_imagenet_webdataset_loaders

# --- Redefine Stage1MANTiSWrapper used during training for loading checkpoints ---
class Stage1MANTiSWrapper(nn.Module):
    """
    A wrapper for the MANTiS Stage 1 model components.
    This definition must match the one used for training to load the checkpoint correctly.
    """
    def __init__(self, client_params, decoder_params, vib_channels):
        super().__init__()
        self.latent_channels = vib_channels

        # Client / Bottleneck / Decoder
        self.client = MANTiSClient(
            num_tasks=client_params.get('num_tasks', 3),
            latent_channels=vib_channels,
        )
        self.vib = VIBBottleneck(vib_channels)
        self.decoder = FrankenSplitDecoder(
            input_channels=vib_channels,
            output_channels=decoder_params['output_channels'],
        )

        # Identity FiLM (γ=1, β=0) kept as a buffer
        self.register_buffer(
            "identity_film_params",
            torch.cat(
                [torch.ones(vib_channels), torch.zeros(vib_channels)]
            ),
        )
        
    def forward(self, x):
        """
        Forward pass for training, adapted here for conceptual clarity.
        The evaluation script will call the components directly.
        """
        b = x.size(0)
        f_stem = self.client.stem(x)
        identity_film = self.identity_film_params.unsqueeze(0).expand(b, -1)
        z_raw = self.client.filmed_encoder(f_stem, identity_film)
        z_hat, z_liks = self.vib(z_raw, training=False)
        recon = self.decoder(z_hat)          # not z_hat_int
        recon = F.adaptive_avg_pool2d(recon, (28, 28))
        z_int  = torch.round(z_hat).to(torch.int16)


        
        return {
            "g_s_output": recon,
            "z_likelihoods": {"z": z_liks},
            "z_raw": z_raw,
        }

# --- Main Evaluation Components ---

class Stage1DownstreamEvaluator(nn.Module):
    """
    End-to-end model for evaluating Stage 1 on a downstream task.
    Combines the trained client/decoder with a pretrained classification tail.
    """
    def __init__(self, stage1_model, downstream_tail):
        super().__init__()
        self.stage1_model = stage1_model
        self.tail = downstream_tail
        self.tail.eval()

    def forward(self, x):
        b = x.size(0)
        
        # 1. Stem -> Encoder (with identity FiLM) -> z_raw
        f_stem = self.stage1_model.client.stem(x)
        identity = self.stage1_model.identity_film_params[None, :].expand(b, -1)
        z_raw = self.stage1_model.client.filmed_encoder(f_stem, identity)
        
        # --- VIB (inference) --------------------------------------------------
        z_hat, z_liks = self.stage1_model.vib(z_raw, training=False)
        z_hat_int = torch.round(z_hat)           # integer latents for EB
        # ----------------------------------------------------------------------
        
        # 3. Decoder -> Reconstructed Features
        recon = self.stage1_model.decoder(z_hat_int)
        recon = F.adaptive_avg_pool2d(recon, (28, 28))
        
        # 4. Pretrained Tail -> Logits
        logits = self.tail(recon)
        
        return logits, z_hat_int, z_liks

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f'top{k}'] = correct_k.mul_(100.0 / batch_size)
        return res

def evaluate(model, val_loader, device, image_size):
    """Run evaluation loop to compute accuracy and BPP."""
    model.eval()
    
    total_samples = 0
    avg_top1 = 0
    avg_top5 = 0
    avg_bpp = 0
    
    pbar = tqdm(val_loader, desc='Evaluating', unit='batch')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            logits, z_hat_int, z_liks = model(images)
            
            # --- 1. Accuracy Calculation ---
            acc_metrics = accuracy(logits, labels, topk=(1, 5))
            avg_top1 += acc_metrics['top1'].item() * batch_size
            avg_top5 += acc_metrics['top5'].item() * batch_size

            # --- 2. BPP Calculation ---
            # Update the entropy bottleneck model to get the latest CDFs
            model.stage1_model.vib.entropy_bottleneck.update(force=False)
            
            # Compress the latents to bitstreams
            compressed_strings = model.stage1_model.vib.entropy_bottleneck.compress(z_hat_int)
            
            # Calculate total bits for the batch
            total_bits = sum(len(s) for s in compressed_strings) * 8
            
            # Calculate BPP (bits per original image pixel)
            bpp = total_bits / (batch_size * image_size * image_size)
            avg_bpp += bpp * batch_size
            
            
            total_samples += batch_size
            pbar.set_postfix({
                'Top-1': f'{avg_top1 / total_samples:.2f}%',
                'BPP': f'{avg_bpp / total_samples:.4f}'
            })
            

    # Final averages
    final_top1 = avg_top1 / total_samples
    final_top5 = avg_top5 / total_samples
    final_bpp = avg_bpp / total_samples
    
    return final_top1, final_top5, final_bpp

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 MANTiS checkpoint for downstream accuracy and BPP.")
    parser.add_argument('--config', type=str, required=True, help="Path to the training YAML config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained Stage 1 .pth checkpoint file.")
    parser.add_argument('--data-dir', type=str, default='~/imagenet-1k-wds', help="Path to ImageNet WebDataset shards.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of data loader workers.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    # --- Load Config ---
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
    print(f"Loaded configuration from: {args.config}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Trained Stage 1 Model ---
    model_cfg = config.model.config
    stage1_model = Stage1MANTiSWrapper(
        client_params=vars(model_cfg.client_params),
        decoder_params=vars(model_cfg.decoder_params),
        vib_channels=model_cfg.vib_channels
    ).to(device)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Stage 1 model weights loaded successfully.")

    # --- 2. Create Pretrained Downstream Tail ---
    # The tail expects features that have been decoded to match ResNet's layer2 output (512 channels)
    # The config file confirms the decoder output_channels are 512.
    print("Initializing ResNetCompatibleTail with pretrained ImageNet-1k weights...")
    downstream_tail = ResNetCompatibleTail(
        input_channels=config.model.config.decoder_params.output_channels,
        num_classes=1000, # Full ImageNet-1k classification
        use_pretrained_layers=True
    ).to(device)
    print("✓ Pretrained tail created.")
    
    # --- 3. Combine into a Single Evaluation Model ---
    eval_model = Stage1DownstreamEvaluator(stage1_model, downstream_tail).to(device)
    total_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
    print(f"✓ Evaluation model assembled. Trainable parameters: {total_params:,}")

    # --- 4. Prepare Data Loader ---
    _, val_loader = create_imagenet_webdataset_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=config.data.image_size
    )
    print(f"Using ImageNet validation data from: {args.data_dir}")

    # --- 5. Run Evaluation ---
    top1, top5, bpp = evaluate(eval_model, val_loader, device, config.data.image_size)

    # --- 6. Report Results ---
    print("\n" + "="*50)
    print("        MANTiS Stage 1 Evaluation Results")
    print("="*50)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Top-1 Accuracy: {top1:.2f}%")
    print(f"  Top-5 Accuracy: {top5:.2f}%")
    print(f"  Rate (BPP):     {bpp:.4f} bits/pixel")
    print("="*50)

if __name__ == '__main__':
    main()