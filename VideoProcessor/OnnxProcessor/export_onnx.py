"""
Export trained PyTorch model to ONNX format for faster inference.
ONNX Runtime can fully utilize all CPU cores (bypasses Python GIL).
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VideoProcessor.video_selector_model import FrameClassifierCNN


def export_to_onnx(
    model_path: str,
    output_path: str,
    backbone: str = 'resnet18',
    image_size: int = 224
):
    """Export PyTorch model to ONNX format."""
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = FrameClassifierCNN(num_classes=2, pretrained=False, backbone=backbone)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input with batch > 1 to ensure dynamic batch works
    dummy_input = torch.randn(2, 3, image_size, image_size)
    
    print(f"Exporting to ONNX: {output_path}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Export complete!")
    print(f"  Output: {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model verified successfully!")


if __name__ == "__main__":
    MODEL_PATH = r"E:\0_projects\00_football_system\checkpoints\best_model.pth"
    ONNX_PATH = r"E:\0_projects\00_football_system\checkpoints\model.onnx"
    
    export_to_onnx(
        model_path=MODEL_PATH,
        output_path=ONNX_PATH,
        backbone='resnet18',
        image_size=224  # Keep original size for quality
    )
