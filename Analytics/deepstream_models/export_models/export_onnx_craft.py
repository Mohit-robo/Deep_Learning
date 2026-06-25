import os
import sys
import torch
from collections import OrderedDict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'numberplate_utils'))

from numberplate_utils.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def export_to_onnx(model_path, onnx_output_path):
    """
    Loads PyTorch weights and exports the CRAFT model to ONNX with dynamic input/output axes.
    """
    print(f"[*] Loading PyTorch weights from {model_path}...")
    model = CRAFT()
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict = copyStateDict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input representing a batch of 1 image with 3 channels and size 416x416
    dummy_input = torch.randn(1, 3, 416, 416, dtype=torch.float32)
    
    print(f"[*] Exporting model to ONNX format at {onnx_output_path}...")
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'feature'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 1: 'height_out', 2: 'width_out'},
            'feature': {0: 'batch_size', 2: 'height_feat', 3: 'width_feat'}
        }
    )
    print("[+] ONNX export completed successfully.")
    
    # Verify the exported ONNX model structurally
    try:
        import onnx
        print("[*] Verifying ONNX model structure...")
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model is structurally valid and verified!")
    except ImportError:
        print("[!] 'onnx' library is not installed in the Python environment. Skipping structural verification.")
        print("    You can install it with: pip install onnx")

if __name__ == "__main__":
    weights_dir = "weights"
    model_path = os.path.join(weights_dir, "craft_mlt_25k.pth")
    onnx_output_path = os.path.join(weights_dir, "craft_mlt_25k.onnx")
    
    if not os.path.exists(model_path):
        print(f"[-] Error: Could not find weight file at {model_path}")
        sys.exit(1)
        
    export_to_onnx(model_path, onnx_output_path)
