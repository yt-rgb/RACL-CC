import argparse
from pathlib import Path

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CLIP_CD_LoRA to HuggingFace format')
    parser.add_argument(
        '--input',
        type=str,
        default='/root/autodl-tmp/checkpoints/CLIP_RegionAware/best_model.pth',
        help='Path to CLIP_CD_LoRA checkpoint (best_model.pth)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/root/autodl-tmp/checkpoints/CLIP_RegionAware_merged-2',
        help='Output directory for merged HuggingFace model',
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='/root/autodl-tmp/clip-vit-large-patch14',
        help='Path to base CLIP model (HuggingFace format, should be clip-vit-large-patch14 NOT 336)',
    )
    parser.add_argument(
        '--lora-layers',
        type=str,
        default='4,10,16,22',
        help='LoRA layers (comma-separated)',
    )
    return parser.parse_args()


def merge_lora_weights(base_weight, lora_A_weight, lora_B_weight):
    delta = lora_B_weight @ lora_A_weight
    return base_weight + delta


def convert_checkpoint(args):
    print("=" * 60)
    print("CLIP_CD_LoRA to HuggingFace Converter")
    print("=" * 60)

    lora_layers = [int(x) for x in args.lora_layers.split(',')]
    print(f"LoRA layers: {lora_layers}")

    print(f"\nLoading checkpoint from: {args.input}")
    checkpoint = torch.load(args.input, map_location='cpu')

    print(f"\nCheckpoint keys ({len(checkpoint)} total):")
    for k in sorted(checkpoint.keys())[:20]:
        print(f"  {k}: {checkpoint[k].shape}")
    if len(checkpoint) > 20:
        print(f"  ... and {len(checkpoint) - 20} more keys")

    print(f"\nLoading base model from: {args.base_model}")
    base_model = CLIPVisionModel.from_pretrained(args.base_model)
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model)

    clip_weights = {}
    for k, v in checkpoint.items():
        if k.startswith('clip.clip.'):
            new_key = k.replace('clip.clip.', '')
            clip_weights[new_key] = v

    print(f"\nExtracted {len(clip_weights)} CLIP weights")

    lora_A_weights = {}
    lora_B_weights = {}
    for k, v in checkpoint.items():
        if 'lora_A_list' in k:
            idx = int(k.split('.')[2])
            lora_A_weights[idx] = v
        elif 'lora_B_list' in k:
            idx = int(k.split('.')[2])
            lora_B_weights[idx] = v

    print(f"Found {len(lora_A_weights)} LoRA A weights")
    print(f"Found {len(lora_B_weights)} LoRA B weights")
    print("\nMerging LoRA weights...")

    merged_state_dict = clip_weights.copy()

    for i, layer_idx in enumerate(lora_layers):
        q_lora_idx = i * 2
        v_lora_idx = i * 2 + 1

        q_proj_key = f'vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight'
        v_proj_key = f'vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight'

        if q_lora_idx in lora_A_weights and q_lora_idx in lora_B_weights:
            orig_q_key = f'vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.original_linear.weight'

            if orig_q_key in clip_weights:
                base_q = clip_weights[orig_q_key]
                lora_A = lora_A_weights[q_lora_idx]
                lora_B = lora_B_weights[q_lora_idx]

                merged_q = merge_lora_weights(base_q, lora_A, lora_B)
                merged_state_dict[q_proj_key] = merged_q

                if orig_q_key in merged_state_dict:
                    del merged_state_dict[orig_q_key]

                print(f"  Merged LoRA for layer {layer_idx} q_proj")

        if v_lora_idx in lora_A_weights and v_lora_idx in lora_B_weights:
            orig_v_key = f'vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.original_linear.weight'

            if orig_v_key in clip_weights:
                base_v = clip_weights[orig_v_key]
                lora_A = lora_A_weights[v_lora_idx]
                lora_B = lora_B_weights[v_lora_idx]

                merged_v = merge_lora_weights(base_v, lora_A, lora_B)
                merged_state_dict[v_proj_key] = merged_v

                if orig_v_key in merged_state_dict:
                    del merged_state_dict[orig_v_key]

                print(f"  Merged LoRA for layer {layer_idx} v_proj")

    keys_to_remove = [
        k for k in merged_state_dict if 'lora_A' in k or 'lora_B' in k or 'original_linear' in k
    ]
    for k in keys_to_remove:
        del merged_state_dict[k]

    print(f"\nFinal merged state_dict has {len(merged_state_dict)} keys")

    missing_keys, unexpected_keys = base_model.load_state_dict(merged_state_dict, strict=False)

    if missing_keys:
        print(f"\nMissing keys: {len(missing_keys)}")
        for k in missing_keys[:10]:
            print(f"  {k}")

    if unexpected_keys:
        print(f"\nUnexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:10]:
            print(f"  {k}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged model to: {output_dir}")
    base_model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("\nYou can now use the merged model in RACL training:")
    print(f'  VISION_TOWER="{output_dir}"')


if __name__ == '__main__':
    args = parse_args()
    convert_checkpoint(args)
