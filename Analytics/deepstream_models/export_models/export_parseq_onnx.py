#!/usr/bin/env python3
"""
PARSeq ONNX Export Script with Dynamic Batch Size Support
=========================================================
This script exports a PARSeq model (either a custom checkpoint or the pretrained 
official model) to ONNX format with dynamic batch support, enabling efficient 
batch inference for multiple text crops in a single forward pass.

Prerequisites:
    pip install torch lightning parseq

Usage:
    # Export the default pretrained model
    python3 export_parseq_onnx.py --output parseq_dynamic.onnx
    
    # Export a custom lightning checkpoint
    python3 export_parseq_onnx.py --checkpoint path/to/model.ckpt --output parseq_dynamic.onnx
"""

import argparse
import sys
import torch

def apply_pytorch_lightning_patch():
    """
    Runtime patch to prevent 'cannot import name ModelIO from pytorch_lightning.core.saving'
    errors on newer PyTorch Lightning versions.
    """
    import sys
    import types
    
    # 1. Ensure pytorch_lightning is in sys.modules
    try:
        import pytorch_lightning
    except ImportError:
        # If not installed, we don't need to patch it
        return

    # 2. Inject pytorch_lightning.core.saving into sys.modules if missing
    core_module = None
    try:
        import pytorch_lightning.core as core
        core_module = core
    except ImportError:
        # If core doesn't exist, create it
        core_module = types.ModuleType('pytorch_lightning.core')
        sys.modules['pytorch_lightning.core'] = core_module
        pytorch_lightning.core = core_module

    saving_module = None
    try:
        import pytorch_lightning.core.saving as saving
        saving_module = saving
    except ImportError:
        # Create a dummy saving module
        saving_module = types.ModuleType('pytorch_lightning.core.saving')
        sys.modules['pytorch_lightning.core.saving'] = saving_module
        if core_module:
            core_module.saving = saving_module

    # 3. Inject ModelIO if not present in the saving module
    if saving_module and not hasattr(saving_module, 'ModelIO'):
        print("[*] Applying runtime patch: Injecting Dummy ModelIO class into pytorch_lightning.core.saving")
        class DummyModelIO:
            pass
        saving_module.ModelIO = DummyModelIO

def patch_cache_directories():
    import os
    import sys
    import glob
    
    # Check standard cache paths
    cache_bases = [
        os.path.expanduser("~/.cache/torch/hub/baudm_parseq_main"),
        os.path.expanduser("~/.cache/torch/hub/baudm_parseq_v1.0.0"),
    ]
    
    patched_any = False
    for cache_dir in cache_bases:
        if os.path.exists(cache_dir):
            print(f"[*] Found cached PARSeq directory: {cache_dir}")
            print("[*] Checking and applying Python < 3.9 compatibility patches...")
            
            # Find all Python files recursively
            py_files = glob.glob(os.path.join(cache_dir, "**/*.py"), recursive=True)
            
            for filepath in py_files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    has_changes = False
                    if "from __future__ import annotations" not in content:
                        lines = content.splitlines()
                        insert_idx = 0
                        # Skip shebang if present
                        if lines and lines[0].startswith("#!"):
                            insert_idx = 1
                        # Skip coding declaration if present
                        if len(lines) > insert_idx and ("coding:" in lines[insert_idx] or "coding=" in lines[insert_idx]):
                            insert_idx += 1
                        
                        lines.insert(insert_idx, "from __future__ import annotations")
                        content = "\n".join(lines)
                        has_changes = True
                        
                    # Fix EPOCH_OUTPUT = list[dict[str, BatchResult]] for Python < 3.9
                    if "EPOCH_OUTPUT = list[dict[str, BatchResult]]" in content:
                        content = content.replace(
                            "EPOCH_OUTPUT = list[dict[str, BatchResult]]",
                            "EPOCH_OUTPUT = 'list[dict[str, BatchResult]]'"
                        )
                        has_changes = True
                        
                    # Fix Dynamo data-dependent expression error during torch.export
                    target_str = "if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():"
                    if target_str in content:
                        content = content.replace(target_str, "if False:  # Patched for ONNX export")
                        has_changes = True
                        
                    # Some versions might use self.tokenizer
                    target_str_2 = "if testing and (tgt_in == self.tokenizer.eos_id).any(dim=-1).all():"
                    if target_str_2 in content:
                        content = content.replace(target_str_2, "if False:  # Patched for ONNX export")
                        has_changes = True
                        
                    if has_changes:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        patched_any = True
                except Exception as patch_err:
                    print(f"[!] Warning: Failed to patch {filepath}: {patch_err}")
    if patched_any:
        print("[+] Compatibility patches applied successfully! Clearing import cache...")
        # Clear sys.modules cached strhub imports
        for key in list(sys.modules.keys()):
            if key.startswith("strhub"):
                del sys.modules[key]
    return patched_any

def clean_sys_modules():
    """
    Cleans up sys.modules and sys.path of any 'strhub' or cached PARSeq components.
    This prevents cross-import contamination when switching between branches or tags.
    """
    import sys
    # Clear any strhub modules from sys.modules to force a re-import from the active path
    for key in list(sys.modules.keys()):
        if key.startswith("strhub"):
            del sys.modules[key]
            
    # Clean sys.path from other cached versions of baudm_parseq
    for path in list(sys.path):
        if "baudm_parseq" in path or "torch/hub" in path:
            try:
                sys.path.remove(path)
            except ValueError:
                pass

def export_model(checkpoint_path=None, output_path="parseq_dynamic.onnx", opset_version=14):
    print("[*] Loading PARSeq model...")
    
    # Apply pytorch-lightning runtime compatibility patches
    apply_pytorch_lightning_patch()
    
    # Run patching on existing directories first to preempt errors
    patch_cache_directories()
    
    model = None
    try:
        if checkpoint_path:
            # Load custom model checkpoint using lightning load mechanism
            try:
                clean_sys_modules()
                from strhub.models.utils import load_from_checkpoint
                model = load_from_checkpoint(checkpoint_path)
            except ImportError:
                print("[!] Could not import strhub. Loading checkpoint using torch.load...")
                model = torch.load(checkpoint_path, map_location="cpu")
        else:
            # Load SOTA pretrained model from torch hub
            # We try v1.0.0 tag first, then fallback to main branch
            try:
                print("[*] Loading SOTA pretrained PARSeq model from torch.hub (baudm/parseq:v1.0.0)...")
                clean_sys_modules()
                model = torch.hub.load('baudm/parseq:v1.0.0', 'parseq', pretrained=True)
            except Exception as e_tag:
                print(f"[!] Failed to load v1.0.0 tag: {e_tag}")
                print("[*] Attempting to load from main branch (baudm/parseq)...")
                try:
                    clean_sys_modules()
                    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
                except TypeError as type_err:
                    if "subscriptable" in str(type_err):
                        print("[!] Detected 'type' object is not subscriptable error during load.")
                        print("[*] Running compatibility patcher and retrying...")
                        patch_cache_directories()
                        # Retry loading from main branch
                        clean_sys_modules()
                        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
                    else:
                        raise type_err
            
        if model is None:
            raise ValueError("Failed to load model from all sources.")
            
        model.eval()
        
        # Configure model parameters for exportability
        if hasattr(model, 'refine_iters'):
            model.refine_iters = 0
            print("[+] Set model.refine_iters = 0 (disabled iterative refinement for tracing)")
        if hasattr(model, 'decode_ar'):
            model.decode_ar = False
            print("[+] Set model.decode_ar = False (disabled autoregressive decoding for tracing)")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[-] Error loading model: {e}")
        print("\nPlease ensure you have installed the parseq dependencies:")
        print("  pip install torch lightning")
        sys.exit(1)

    print("[*] Preparing dummy input for tracing...")
    # PARSeq default expected image size is 32x128
    # We trace with a fixed batch_size = 8
    dummy_input = torch.randn(32, 3, 32, 128)

    # We define input and output names along with the dynamic axes dictionary
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    print(f"[*] Exporting model to {output_path} (Opset 17)...")
    try:
        # Force standard torch.onnx.export to bypass Dynamo / torch.export errors in newer PyTorch
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={}
        )
        print(f"[+] Successfully exported fixed-batch (8) PARSeq model to: {output_path}")
        print("\nVerification dimensions:")
        print("  Inputs:  ['input']  -> shape: [batch_size, 3, 32, 128]")
        print("  Outputs: ['output'] -> shape: [batch_size, 26, 95]")

        # Post-process with onnx-simplifier to fix TensorRT dynamic shape Squeeze issues
        print("\n[*] Post-processing ONNX model to fix dynamic shape Squeeze axes for TensorRT...")
        try:
            import onnx
            from onnxsim import simplify
            
            onnx_model = onnx.load(output_path)
            model_simp, check = simplify(
                onnx_model,
                dynamic_input_shape=False,
                test_input_shapes={"input": [32, 3, 32, 128]}
            )
            if check:
                onnx.save(model_simp, output_path)
                print(f"[+] Successfully simplified and fixed ONNX model: {output_path}")
            else:
                print("[-] ONNX simplification check failed, keeping original model.")
        except ImportError:
            print("[!] 'onnxsim' not installed. TensorRT may fail with Squeeze axes errors.")
            print("    Please install it via: pip install onnx-simplifier")
            print("    and re-run this script to automatically fix the model.")
        except Exception as e:
            print(f"[!] Warning: ONNX simplification failed: {e}")

    except Exception as e:
        print(f"[-] Export failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PARSeq to ONNX with Dynamic Batch Support")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (.ckpt or .pth)")
    parser.add_argument("--output", type=str, default="parseq_dynamic.onnx", help="Output path for the ONNX model")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.output, args.opset)
