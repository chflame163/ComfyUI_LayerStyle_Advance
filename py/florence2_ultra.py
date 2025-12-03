# layerstyle advance

import os
import sys

# CRITICAL: Set environment BEFORE any transformers import
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Aggressively patch transformers to skip docstring validation
# This MUST happen before any transformers imports
import importlib
import builtins

_original_import = builtins.__import__

def _patched_import(name, *args, **kwargs):
    module = _original_import(name, *args, **kwargs)
    
    # Patch ModelOutput when ANY transformers module is imported
    if 'transformers' in name:
        try:
            # Try to find ModelOutput in this module or its submodules
            if hasattr(module, 'ModelOutput'):
                _patch_model_output(module.ModelOutput)
            # Also check utils.generic
            if hasattr(module, 'utils'):
                utils = getattr(module, 'utils', None)
                if utils and hasattr(utils, 'generic'):
                    generic = getattr(utils, 'generic', None)
                    if generic and hasattr(generic, 'ModelOutput'):
                        _patch_model_output(generic.ModelOutput)
        except:
            pass
    
    return module

def _patch_model_output(ModelOutput):
    """Patch ModelOutput to skip docstring validation"""
    if hasattr(ModelOutput, '__init_subclass__'):
        original = ModelOutput.__init_subclass__
        @classmethod
        def safe_init_subclass(cls, **kwargs):
            try:
                return original.__func__(cls, **kwargs)
            except ValueError as e:
                if "Args" in str(e) or "Parameters" in str(e):
                    return None  # Skip docstring validation
                raise
        ModelOutput.__init_subclass__ = safe_init_subclass

builtins.__import__ = _patched_import

import io
import torch
from unittest.mock import patch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys
from transformers.dynamic_module_utils import get_imports
from transformers import PreTrainedModel

# CRITICAL: Patch transformers docstring validation functions
try:
    from transformers.utils.generic import ModelOutput
    if hasattr(ModelOutput, '__init_subclass__'):
        original_init_subclass = ModelOutput.__init_subclass__
        @classmethod
        def safe_init_subclass(cls, **kwargs):
            try:
                return original_init_subclass.__func__(cls, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "Args" in error_str or "Parameters" in error_str or "docstring" in error_str.lower():
                    return None
                raise
        ModelOutput.__init_subclass__ = safe_init_subclass
        print("[FLORENCE2] Patched ModelOutput.__init_subclass__")
except Exception as e:
    print(f"[FLORENCE2] Could not patch ModelOutput: {e}")

# CRITICAL: Patch _prepare_output_docstrings which is called by decorators
try:
    from transformers.utils import doc as doc_utils
    if hasattr(doc_utils, '_prepare_output_docstrings'):
        original_prepare = doc_utils._prepare_output_docstrings
        def patched_prepare_output_docstrings(output_type, config_class, min_indent=0):
            try:
                return original_prepare(output_type, config_class, min_indent)
            except ValueError as e:
                error_str = str(e)
                if "Args" in error_str or "Parameters" in error_str:
                    # Return empty docstring instead of raising
                    return ""
                raise
        doc_utils._prepare_output_docstrings = patched_prepare_output_docstrings
        print("[FLORENCE2] Patched _prepare_output_docstrings")
except Exception as e:
    print(f"[FLORENCE2] Could not patch _prepare_output_docstrings: {e}")

import comfy.model_management
from .imagefunc import *

# Disable CUDA SDPA backends
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False) 
    torch.backends.cuda.enable_math_sdp(True)
except:
    pass

# Monkey-patch the _check_and_enable_sdpa function in transformers
try:
    import transformers.modeling_utils as modeling_utils
    original_from_pretrained = modeling_utils.PreTrainedModel.from_pretrained.__func__
    
    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        # Force eager attention for Florence2 models
        if 'florence' in str(args[0]).lower() if args else False:
            kwargs['attn_implementation'] = None
            # Set _supports_sdpa on the class before loading
            if not hasattr(cls, '_supports_sdpa'):
                cls._supports_sdpa = False
                cls._supports_flash_attn_2 = False
        return original_from_pretrained(cls, *args, **kwargs)
    
    # Don't apply this global patch - too risky
    # modeling_utils.PreTrainedModel.from_pretrained = patched_from_pretrained
except Exception as e:
    pass

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

device = comfy.model_management.get_torch_device()

def patch_transformers_for_florence2():
    """
    Patches for Transformers >=4.42 / 5.x compatibility with Florence2 models.
    Based on: https://github.com/kijai/ComfyUI-Florence2/issues/184
    """
    patched = []
    
    # Patch 1: _tie_or_clone_weights was removed in newer transformers
    if not hasattr(PreTrainedModel, '_tie_or_clone_weights'):
        def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
            """Tie or clone module weights depending on whether we are using TorchScript or not"""
            output_embeddings.weight = input_embeddings.weight
            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = torch.nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings
        
        PreTrainedModel._tie_or_clone_weights = _tie_or_clone_weights
        patched.append("_tie_or_clone_weights")
    
    # Patch 2: _supports_sdpa attribute check changed in newer transformers
    if not hasattr(PreTrainedModel, '_supports_sdpa'):
        PreTrainedModel._supports_sdpa = True
        patched.append("_supports_sdpa")
    
    if patched:
        log(f"Applied Florence2 compatibility patches: {', '.join(patched)}")

# Apply patches at module load time
patch_transformers_for_florence2()

fl2_model_repos = {
    "base": "microsoft/Florence-2-base",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large": "microsoft/Florence-2-large",
    "large-ft": "microsoft/Florence-2-large-ft",
    "DocVQA": "HuggingFaceM4/Florence-2-DocVQA",
    "SD3-Captioner": "gokaygokay/Florence-2-SD3-Captioner",
    "base-PromptGen": "MiaoshouAI/Florence-2-base-PromptGen",
    "CogFlorence-2-Large-Freeze": "thwri/CogFlorence-2-Large-Freeze",
    "CogFlorence-2.1-Large": "thwri/CogFlorence-2.1-Large",
    "base-PromptGen-v1.5":"MiaoshouAI/Florence-2-base-PromptGen-v1.5",
    "large-PromptGen-v1.5":"MiaoshouAI/Florence-2-large-PromptGen-v1.5",
    "base-PromptGen-v2.0":"MiaoshouAI/Florence-2-base-PromptGen-v2.0",
    "large-PromptGen-v2.0":"MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    "Florence-2-Flux":"gokaygokay/Florence-2-Flux",
    "Florence-2-Flux-Large":"gokaygokay/Florence-2-Flux-Large"
}

def fixed_get_imports(filename) -> list[str]:
    """Workaround for FlashAttention"""
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        pass
    return imports

def patch_florence2_model_file(model_path):
    """
    Patch modeling_florence2.py for newer transformers compatibility.
    FORCE patch - adds _supports_sdpa = False to ALL classes
    
    NOTE: If transformers >= 5.x fixes Florence2 compatibility (see issue #39974),
    consider updating transformers instead of using these patches.
    These patches are minimal and only fix essential compatibility issues:
    - _supports_sdpa attribute
    - _tie_or_clone_weights method
    - ModelOutput docstring validation
    """
    # Patch BOTH the model file AND the transformers cache
    files_to_patch = [os.path.join(model_path, "modeling_florence2.py")]
    
    # Also find and patch cached versions in huggingface cache
    import glob
    hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.exists(hf_cache):
        cached_files = glob.glob(f"{hf_cache}/**/modeling_florence2.py", recursive=True)
        files_to_patch.extend(cached_files)
        log(f"[PATCH] Found {len(cached_files)} cached Florence2 files")
    
    patched_count = 0
    for modeling_file in files_to_patch:
        if _patch_single_file(modeling_file):
            patched_count += 1
    
    log(f"[PATCH] Patched {patched_count}/{len(files_to_patch)} files")
    return patched_count > 0

def _patch_single_file(modeling_file):
    """Patch a single modeling_florence2.py file"""
    log(f"[PATCH] Checking {modeling_file}")
    
    if not os.path.exists(modeling_file):
        log(f"[PATCH] File not found!")
        return False
    
    with open(modeling_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    log(f"[PATCH] File size: {len(content)} bytes")
    log(f"[PATCH] Already patched: {'PATCHED_FOR_NEW_TRANSFORMERS' in content}")
    log(f"[PATCH] Has _supports_sdpa: {'_supports_sdpa' in content}")
    log(f"[PATCH] Has _tie_or_clone_weights: {'_tie_or_clone_weights' in content}")
    
    if '# PATCHED_FOR_NEW_TRANSFORMERS' in content:
        log("[PATCH] Already patched, skipping")
        return True
    
    # Force add _supports_sdpa to ALL Florence2 classes
    lines = content.split('\n')
    new_lines = ['# PATCHED_FOR_NEW_TRANSFORMERS']
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        # After any class definition that contains "Florence2", add the attributes
        if line.strip().startswith('class ') and 'Florence2' in line and line.strip().endswith(':'):
            log(f"[PATCH] Found class at line {i+1}: {line.strip()}")
            new_lines.append('    _supports_sdpa = False')
            new_lines.append('    _supports_flash_attn_2 = False')
    
    # Replace _tie_or_clone_weights
    content = '\n'.join(new_lines)
    if '_tie_or_clone_weights' in content:
        import re
        old_count = content.count('_tie_or_clone_weights')
        content = re.sub(
            r'self\._tie_or_clone_weights\(([^,]+),\s*([^)]+)\)',
            r'\1.weight = \2.weight',
            content
        )
        new_count = content.count('_tie_or_clone_weights')
        log(f"[PATCH] Replaced _tie_or_clone_weights: {old_count} -> {new_count}")
    
    # PATCH: Fix ALL past_key_values[0][0].shape accesses to handle None
    # There are multiple places in the code that access past_key_values[0][0].shape
    # We need to replace ALL of them with safe versions
    
    # Pattern 1: past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    # This pattern checks if past_key_values is not None but doesn't check if past_key_values[0][0] is None
    old_pattern1 = r'past_key_values_length = past_key_values\[0\]\[0\]\.shape\[2\] if past_key_values is not None else 0'
    new_pattern1 = 'past_key_values_length = past_key_values[0][0].shape[2] if (past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None) else 0'
    if re.search(old_pattern1, content):
        content = re.sub(old_pattern1, new_pattern1, content)
        log("[PATCH] ✓ Patched past_key_values_length assignments")
    
    # Pattern 2: past_length = past_key_values[0][0].shape[2] (standalone)
    old_pattern2 = r'(\s+)past_length = past_key_values\[0\]\[0\]\.shape\[2\](\s*\n)'
    def replace_past_length(match):
        indent = match.group(1)
        newline = match.group(2)
        return f'{indent}past_length = past_key_values[0][0].shape[2] if (past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None) else 0{newline}'
    if re.search(old_pattern2, content):
        content = re.sub(old_pattern2, replace_past_length, content)
        log("[PATCH] ✓ Patched past_length assignments")
    
    # NOTE: Removed complex generate() patches - they were causing issues
    # If transformers >= 5.x fixes Florence2, update transformers instead
    # Keeping only essential patches: _supports_sdpa, _tie_or_clone_weights, ModelOutput
    
    # CRITICAL FIX: Replace ModelOutput inheritance to avoid docstring validation
    import re
    
    log(f"[PATCH] Checking for ModelOutput inheritance...")
    model_output_matches = re.findall(r'class (\w+Output)\(ModelOutput\)', content)
    log(f"[PATCH] Found {len(model_output_matches)} classes inheriting from ModelOutput: {model_output_matches}")
    
    # Check if file uses ModelOutput (must contain the inheritance pattern)
    if re.search(r'class \w+Output\(ModelOutput\)', content):
        log("[PATCH] ✓ Found ModelOutput inheritance, replacing...")
        
        # Add our simple dataclass at the very beginning of the file
        import_section = """# PATCHED: Simple output class to bypass ModelOutput docstring validation
from dataclasses import dataclass
from typing import Optional, Tuple, Any

@dataclass 
class Florence2SimpleOutput:
    \"\"\"
    Simple output class for Florence2 models.
    
    Args:
        loss: Optional loss tensor.
        logits: Model logits.
        last_hidden_state: Last hidden state.
        past_key_values: Past key values for caching.
        decoder_hidden_states: Decoder hidden states.
        decoder_attentions: Decoder attention weights.
        cross_attentions: Cross attention weights.
        encoder_last_hidden_state: Encoder last hidden state.
        encoder_hidden_states: Encoder hidden states.
        encoder_attentions: Encoder attention weights.
        image_hidden_states: Image hidden states.
    \"\"\"
    loss: Any = None
    logits: Any = None
    last_hidden_state: Any = None
    past_key_values: Any = None
    decoder_hidden_states: Any = None
    decoder_attentions: Any = None
    cross_attentions: Any = None
    encoder_last_hidden_state: Any = None
    encoder_hidden_states: Any = None
    encoder_attentions: Any = None
    image_hidden_states: Any = None

"""
        # Add at the beginning, after the first line if it's a comment
        if content.startswith('#'):
            first_newline = content.find('\n')
            content = content[:first_newline+1] + import_section + content[first_newline+1:]
        else:
            content = import_section + content
        
        # Replace ALL ModelOutput inheritance with Florence2SimpleOutput
        before_replace = content.count('(ModelOutput)')
        content = re.sub(
            r'class (\w+)\(ModelOutput\)',
            r'class \1(Florence2SimpleOutput)',
            content
        )
        after_replace = content.count('(ModelOutput)')
        florence2_simple_count = content.count('(Florence2SimpleOutput)')
        log(f"[PATCH] ✓ Replaced ModelOutput with Florence2SimpleOutput")
        log(f"[PATCH] Before: {before_replace} ModelOutput, After: {after_replace} ModelOutput, {florence2_simple_count} Florence2SimpleOutput")
    
    # Replace imports of problematic decorators with no-op versions
    # Find where decorators are imported from transformers
    decorator_imports = [
        r'from transformers\.utils\.doc import (replace_return_docstrings|add_start_docstrings_to_model_forward|add_start_docstrings|add_end_docstrings)',
        r'from transformers\.utils import (replace_return_docstrings|add_start_docstrings_to_model_forward|add_start_docstrings|add_end_docstrings)',
    ]
    
    for pattern in decorator_imports:
        if re.search(pattern, content):
            # Replace the import with our no-op versions
            noop_defs = '''
# PATCHED: No-op decorators to bypass docstring validation
def _noop_decorator(*args, **kwargs):
    def decorator(fn):
        return fn
    if args and callable(args[0]):
        return args[0]
    return decorator

replace_return_docstrings = _noop_decorator
add_start_docstrings_to_model_forward = _noop_decorator  
add_start_docstrings = _noop_decorator
add_end_docstrings = _noop_decorator
'''
            # Replace the import line
            content = re.sub(pattern, '# PATCHED: removed decorator import', content)
            # Add no-op definitions right after Florence2SimpleOutput class
            if 'class Florence2SimpleOutput' in content:
                class_end = content.find('\nclass ', content.find('class Florence2SimpleOutput'))
                if class_end > 0:
                    content = content[:class_end] + '\n' + noop_defs + content[class_end:]
                    log("[PATCH] Replaced decorator imports with no-op versions")
                    break
    
    # Validate Python syntax before writing
    log(f"[PATCH] Validating Python syntax...")
    try:
        compile(content, modeling_file, 'exec')
        log(f"[PATCH] ✓ Syntax validation passed")
    except SyntaxError as e:
        log(f"[PATCH] ✗ Syntax error after patching: {e} at line {e.lineno}", message_type='error')
        log(f"[PATCH] Error text: {e.text}")
        # Try to fix common issues
        # Remove any double newlines
        content = re.sub(r'\n\n\n+', '\n\n', content)
        log(f"[PATCH] Trying to fix syntax errors...")
        # Try again
        try:
            compile(content, modeling_file, 'exec')
            log(f"[PATCH] ✓ Syntax fixed!")
        except SyntaxError as e2:
            log(f"[PATCH] ✗ Still has syntax error: {e2} at line {e2.lineno}", message_type='error')
            log(f"[PATCH] Error text: {e2.text}")
            return False
    
    # Write the patched file
    log(f"[PATCH] Writing patched file to {modeling_file}...")
    with open(modeling_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log(f"[PATCH] ✓ File patched successfully! ({len(content)} bytes)")
    return True

def load_model(version):
    florence_path = os.path.join(folder_paths.models_dir, "florence2")
    os.makedirs(florence_path, exist_ok=True)

    model_path = os.path.join(florence_path, version)

    if not os.path.exists(model_path):
        log(f"Downloading Florence2 {version} model...")
        repo_id = fl2_model_repos[version]
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])
    
    # Patch the model file for newer transformers compatibility
    patch_florence2_model_file(model_path)

    # Clear cached Florence2 modules to ensure patched file is used
    import sys
    modules_to_remove = [key for key in list(sys.modules.keys()) if 'florence' in key.lower()]
    for mod in modules_to_remove:
        try:
            del sys.modules[mod]
            log(f"[LOAD] Cleared cached module: {mod}")
        except:
            pass
    
    # Also clear transformers module cache for this model
    cache_key = f"transformers_modules.{os.path.basename(model_path)}"
    modules_to_remove = [key for key in list(sys.modules.keys()) if cache_key in key]
    for mod in modules_to_remove:
        try:
            del sys.modules[mod]
            log(f"[LOAD] Cleared transformers cache: {mod}")
        except:
            pass
    
    log(f"[LOAD] Loading model from {model_path}")
    
    # Re-patch cache files that might have been created after initial patch
    import glob
    hf_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    log(f"[LOAD] Checking HuggingFace cache at: {hf_cache}")
    log(f"[LOAD] Cache exists: {os.path.exists(hf_cache)}")
    
    if os.path.exists(hf_cache):
        cached_files = glob.glob(f"{hf_cache}/**/modeling_florence2.py", recursive=True)
        log(f"[LOAD] Found {len(cached_files)} cached Florence2 files")
        for cached_file in cached_files:
            log(f"[LOAD] Re-patching cache file: {cached_file}")
            _patch_single_file(cached_file)
            log(f"[LOAD] ✓ Re-patched cache file: {cached_file}")
    
    # CRITICAL: Patch _prepare_output_docstrings which is called during module loading
    log("[LOAD] Patching transformers docstring validation...")
    try:
        from transformers.utils import doc as doc_utils
        log(f"[LOAD] doc_utils imported: {doc_utils}")
        
        if hasattr(doc_utils, '_prepare_output_docstrings'):
            original_prepare = doc_utils._prepare_output_docstrings
            log(f"[LOAD] Original _prepare_output_docstrings: {original_prepare}")
            
            def patched_prepare_output_docstrings(output_type, config_class, min_indent=0):
                try:
                    log(f"[LOAD] _prepare_output_docstrings called for: {output_type}")
                    result = original_prepare(output_type, config_class, min_indent)
                    log(f"[LOAD] _prepare_output_docstrings succeeded for: {output_type}")
                    return result
                except ValueError as e:
                    error_str = str(e)
                    log(f"[LOAD] _prepare_output_docstrings exception: {error_str}")
                    if "Args" in error_str or "Parameters" in error_str:
                        log(f"[LOAD] ✓ Suppressed docstring error, returning empty string")
                        return ""
                    log(f"[LOAD] ✗ Re-raising exception")
                    raise
            
            doc_utils._prepare_output_docstrings = patched_prepare_output_docstrings
            log("[LOAD] ✓ Patched _prepare_output_docstrings")
        else:
            log("[LOAD] ✗ doc_utils has no _prepare_output_docstrings")
    except Exception as e:
        import traceback
        log(f"[LOAD] ✗ Could not patch _prepare_output_docstrings: {e}")
        log(f"[LOAD] Traceback: {traceback.format_exc()}")
    
    # Load the model
    log(f"[LOAD] Starting model load from {model_path}")
    
    try:
        log("[LOAD] Calling AutoModelForCausalLM.from_pretrained...")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            log(f"[LOAD] ✓ Model loaded successfully: {type(model).__name__}")
            log(f"[LOAD] Model class: {model.__class__}")
            log(f"[LOAD] Model has _supports_sdpa: {hasattr(model, '_supports_sdpa')}")
            log(f"[LOAD] Model has 'generate': {hasattr(model, 'generate')}")
            log(f"[LOAD] Model has 'language_model': {hasattr(model, 'language_model')}")
            log(f"[LOAD] Model has 'model': {hasattr(model, 'model')}")
            
            # Patch the loaded model class if needed
            if not hasattr(model.__class__, '_supports_sdpa'):
                model.__class__._supports_sdpa = False
                model.__class__._supports_flash_attn_2 = False
                log(f"[LOAD] ✓ Added _supports_sdpa to {model.__class__.__name__}")
            
            # Debug: Check model structure
            if hasattr(model, 'language_model'):
                lang_model = model.language_model
                log(f"[LOAD] language_model type: {type(lang_model)}")
                log(f"[LOAD] language_model has 'generate': {hasattr(lang_model, 'generate')}")
                log(f"[LOAD] language_model attributes: {[a for a in dir(lang_model) if not a.startswith('_')][:10]}")
            if hasattr(model, 'model'):
                sub_model = model.model
                log(f"[LOAD] model type: {type(sub_model)}")
                log(f"[LOAD] model has 'generate': {hasattr(sub_model, 'generate')}")
            
            # Check if Florence2Seq2SeqLMOutput was loaded and patch it
            import sys
            for module_name, module in sys.modules.items():
                if 'florence' in module_name.lower() and hasattr(module, 'Florence2Seq2SeqLMOutput'):
                    output_class = getattr(module, 'Florence2Seq2SeqLMOutput')
                    log(f"[LOAD] Found Florence2Seq2SeqLMOutput in {module_name}")
                    log(f"[LOAD] Output class: {output_class}")
                    log(f"[LOAD] Output class bases: {output_class.__bases__}")
                    # Patch the class to skip docstring validation
                    if hasattr(output_class, '__init_subclass__'):
                        original = output_class.__init_subclass__
                        @classmethod
                        def patched_init_subclass(cls, **kwargs):
                            try:
                                return original.__func__(cls, **kwargs)
                            except Exception as e:
                                log(f"[LOAD] Suppressed docstring error in {cls.__name__}: {e}")
                                return None
                        output_class.__init_subclass__ = patched_init_subclass
                        log(f"[LOAD] ✓ Patched Florence2Seq2SeqLMOutput.__init_subclass__")
            
            log("[LOAD] Calling AutoProcessor.from_pretrained...")
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            log(f"[LOAD] ✓ Processor loaded: {type(processor).__name__}")
            log(f"[LOAD] Processor: {processor}")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"[LOAD] ✗ Error loading model: {str(e)}", message_type='error')
        log(f"[LOAD] Error details:\n{error_details}", message_type='error')
        return (None, None)
    
    log(f"[LOAD] ✓ Successfully loaded model and processor")
    return (model.to(device), processor)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil = Image.open(buf)
    plt.close()
    return pil

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(image.width / 100, image.height / 100)
    ax.imshow(image)
    for i, (bbox, label) in enumerate(zip(data['bboxes'], data['labels'])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        enum_label = f"{i}: {label}"
        plt.text(x1 + 7, y1 + 17, enum_label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def generate_color(index, total_colors=25):
    # Generate color by varying the hue to maximize difference between colors
    hue = (index / total_colors) % 1.0  # Normalize hue to be between 0 and 1
    saturation = 0.65  # Keep saturation constant
    lightness = 0.5  # Keep lightness constant

    # Convert HSL to RGB, then to hexadecimal
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f'#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}'

def plot_mask_bbox(image, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(image.width / 100, image.height / 100)
    ax.imshow(image)
    num_bboxes = len(data['bboxes'])
    for i, (bbox, label) in enumerate(list(zip(data['bboxes'], data['labels']))[1:], start=1):
        x1, y1, x2, y2 = bbox
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        color = generate_color(i, total_colors=num_bboxes)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        enum_label = f"{i}: {label}"
        plt.text(x1 + 7, y1 + 17, enum_label, color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
    ax.axis('off')
    return fig

def plot_mask(image, data, indexes):
    # Create a black background image (mode "1" for binary, "L" for grayscale)
    mask = Image.new("L", (image.width, image.height), 0)  # Black background
    fig, ax = plt.subplots()
    fig.set_size_inches(mask.width / 100, mask.height / 100)
    ax.imshow(mask, cmap='gray')  # Display the mask in grayscale
    ax.set_facecolor('black')  # Set the axes background to black
    fig.patch.set_facecolor('black')  # Set the figure background to black
    for i, (bbox, label) in enumerate(list(zip(data['bboxes'], data['labels']))[1:], start=1):
        x1, y1, x2, y2 = bbox
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1        
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='w', facecolor='w')
        if i in indexes:
            ax.add_patch(rect)
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = color if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return output_image


def convert_to_od_format(data):
    od_results = {
        'bboxes': data.get('bboxes', []),
        'labels': data.get('bboxes_labels', [])
    }
    return od_results


def draw_ocr_bboxes(image, prediction):
    scale = 1
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return output_image


def run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Check model structure and use the correct generate method
    log(f"[RUN] Model type: {type(model)}")
    log(f"[RUN] Model has 'generate': {hasattr(model, 'generate')}")
    
    # Try to use model.generate directly
    if hasattr(model, 'generate'):
        log("[RUN] Using model.generate()")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            do_sample=do_sample,
            num_beams=num_beams,
        )
    elif hasattr(model, 'model') and hasattr(model.model, 'generate'):
        log("[RUN] Using model.model.generate()")
        generated_ids = model.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            do_sample=do_sample,
            num_beams=num_beams,
        )
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'generate'):
        log("[RUN] Using model.language_model.generate()")
        generated_ids = model.language_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            do_sample=do_sample,
            num_beams=num_beams,
        )
    else:
        log("[RUN] ✗ No generate method found!")
        raise AttributeError(f"Model {type(model)} has no generate method")
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def process_image(model, processor, image, task_prompt, max_new_tokens, num_beams, do_sample, fill_mask, text_input=None):
    if task_prompt == 'caption':
        task_prompt = '<CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'detailed caption':
        task_prompt = '<DETAILED_CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'more detailed caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'object detection':
        task_prompt = '<OD>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<OD>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'dense region caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'region proposal':
        task_prompt = '<REGION_PROPOSAL>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'region proposal (mask)':
        task_prompt = '<REGION_PROPOSAL>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        indexes = []
        if isinstance(text_input, str):
            for i in text_input.split(','):
                try:
                    indexes.append(int(i))
                except ValueError:
                    print(f"{i} is nit an instance of int")
        if len(indexes) > 0:
            fig = plot_mask(image, results['<REGION_PROPOSAL>'], indexes)
            pil = fig_to_pil(fig).resize((image.width, image.height), Image.Resampling.LANCZOS)
        else:
            fig = plot_mask_bbox(image, results['<REGION_PROPOSAL>'])
            pil = fig_to_pil(fig)
        return results[task_prompt], pil
    elif task_prompt == 'caption to phrase grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results[task_prompt], fig_to_pil(fig)
    elif task_prompt == 'referring expression segmentation':
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        output_image = draw_polygons(image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask)
        return results[task_prompt], output_image
    elif task_prompt == 'region to segmentation':
        task_prompt = '<REGION_TO_SEGMENTATION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        output_image = draw_polygons(image, results['<REGION_TO_SEGMENTATION>'], fill_mask)
        return results[task_prompt], output_image
    elif task_prompt == 'open vocabulary detection':
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        fig = plot_bbox(image, bbox_results)
        return bbox_results, fig_to_pil(fig)
    elif task_prompt == 'region to category':
        task_prompt = '<REGION_TO_CATEGORY>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        return results[task_prompt], None
    elif task_prompt == 'region to description':
        task_prompt = '<REGION_TO_DESCRIPTION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample, text_input)
        return results[task_prompt], None
    elif task_prompt == 'OCR':
        task_prompt = '<OCR>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'OCR with region':
        task_prompt = '<OCR_WITH_REGION>'
        results = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        output_image = draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])
        output_results = {'bboxes': results[task_prompt].get('quad_boxes', []),
                          'labels': results[task_prompt].get('labels', [])}
        return output_results, output_image
    # gokaygokay/Florence-2-SD3-Captioner task
    elif task_prompt == 'description':
        task_prompt = '<DESCRIPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    # MiaoshouAI/Florence-2-large-PromptGen-v1.5 task
    elif task_prompt == 'generate tags(PromptGen 1.5)':
        task_prompt = '<GENERATE_TAGS>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'mixed caption(PromptGen 1.5)':
        task_prompt = '<MIXED_CAPTION>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'mixed caption plus(PromptGen 2.0)':
        task_prompt = '<MIXED_CAPTION_PLUS>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None
    elif task_prompt == 'analyze(PromptGen 2.0)':
        task_prompt = '<<ANALYZE>>'
        result = run_example(model, processor, task_prompt, image, max_new_tokens, num_beams, do_sample)
        return result[task_prompt], None

    else:
        return "", None  # Return empty string and None for unknown task prompts


def remove_angle_bracket_content(text):
    import re
    # 正则表达式匹配 "<>" 包围的内容，包括尖括号本身
    pattern = r'<[^>]*>'
    # 使用 re.sub 替换匹配的内容为空字符串
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def decode_f_bboxes(F_BBOXES):
    if isinstance(F_BBOXES, str):
        return (torch.zeros(1, 512, 512, dtype=torch.float32), F_BBOXES)

    width = F_BBOXES["width"]
    height = F_BBOXES["height"]
    mask = np.zeros((height, width), dtype=np.uint8)

    x1_c = width
    y1_c = height
    x2_c = y2_c = 0
    label = ""
    if "bboxes" in F_BBOXES:
        for idx in range(len(F_BBOXES["bboxes"])):
            bbox = F_BBOXES["bboxes"][idx]

            new_label = F_BBOXES["labels"][idx].removeprefix("</s>")
            if new_label not in label:
                if idx > 0:
                    label = label + ", "
                label = label + new_label

            if len(bbox) == 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            elif len(bbox) == 8:
                x1 = int(min(bbox[0::2]))
                x2 = int(max(bbox[0::2]))
                y1 = int(min(bbox[1::2]))
                y2 = int(max(bbox[1::2]))
            else:
                continue

            x1_c = min(x1_c, x1)
            y1_c = min(y1_c, y1)
            x2_c = max(x2_c, x2)
            y2_c = max(y2_c, y2)

            mask[y1:y2, x1:x2] = 1

    else:
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)

        x1_c = width
        y1_c = height
        x2_c = y2_c = 0

        for polygon in F_BBOXES["polygons"][0]:
            _polygon = np.array(polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            draw.polygon(_polygon.flatten().tolist(), outline='white', fill='white')

            x1_c = min(x1_c, int(min(polygon[0::2])))
            x2_c = max(x2_c, int(max(polygon[0::2])))
            y1_c = min(y1_c, int(min(polygon[1::2])))
            y2_c = max(y2_c, int(max(polygon[1::2])))

        mask = np.asarray(image)[..., 0].astype(np.float32) / 255

    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    # label = remove_angle_bracket_content(label)
    return (mask, label)


class LS_LoadFlorence2Model:
    def __init__(self):
        self.model = None
        self.processor = None
        self.version = None

    @classmethod
    def INPUT_TYPES(s):
        model_list = list(fl2_model_repos.keys())
        return {
            "required": {
                "version": (model_list,{"default": model_list[0]}),
            },
        }

    RETURN_TYPES = ("FLORENCE2",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "load"
    CATEGORY = '😺dzNodes/LayerMask'

    def load(self, version):
        if self.version != version:
            self.model, self.processor = load_model(version)
            self.version = version

        return ({'model': self.model, 'processor': self.processor, 'version': self.version, 'device': device},)


class Florence2Ultra:
    def __init__(self):
        self.NODE_NAME = 'Florence2Ultra'

    @classmethod
    def INPUT_TYPES(s):
        segment_task_list = [
            "region to segmentation",
            "referring expression segmentation",
            "open vocabulary detection",
            ]
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "florence2_model": ("FLORENCE2",),
                "image": ("IMAGE",),
                "task": (segment_task_list,{"default": segment_task_list[0]}),
                "text_input": ("STRING", {"default": "subject"}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "florence2_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def florence2_ultra(self, florence2_model, image, task, text_input,
                        detail_method, detail_erode, detail_dilate,
                        black_point, white_point, process_detail, device, max_megapixels):
        max_new_tokens = 512
        num_beams = 3
        do_sample = False
        fill_mask = False

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        model = florence2_model['model']
        processor = florence2_model['processor']

        for i in image:
            img = tensor2pil(i).convert("RGB")

            results, _ = process_image(model, processor, img, task,
                                          max_new_tokens, num_beams, do_sample,
                                          fill_mask, text_input)

            if isinstance(results, dict):
                results["width"] = img.width
                results["height"] = img.height

            _mask, _ = decode_f_bboxes(results)

            if process_detail:
                detail_range = detail_erode + detail_dilate
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(img, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = tensor2pil(_mask)

            ret_image = RGB2RGBA(img, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


class Florence2Image2Prompt:

    def __init__(self):
        self.NODE_NAME = 'Florence2Image2Prompt'

    @classmethod
    def INPUT_TYPES(s):
        caption_task_list = [
            "caption",
            "detailed caption",
            "more detailed caption",
            'description',
            'generate tags(PromptGen 1.5)',
            'mixed caption(PromptGen 1.5)',
            'mixed caption plus(PromptGen 2.0)',
            'analyze(PromptGen 2.0)',
            "object detection",
            "dense region caption",
            "region proposal",
            "region proposal (mask)",
            "caption to phrase grounding",
            "open vocabulary detection",
            "region to category",
            "region to description",
            "OCR",
            "OCR with region",
            ]
        return {
            "required": {
                "florence2_model": ("FLORENCE2",),
                "image": ("IMAGE",),
                "task": (caption_task_list,{"default": caption_task_list[2]}),
                "text_input": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 1024, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "step": 1}),
                "do_sample": ('BOOLEAN', {"default": False}),
                "fill_mask": ('BOOLEAN', {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("text", "preview_image",)
    FUNCTION = "florence2_image2prompt"
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def florence2_image2prompt(self, florence2_model, image, task, text_input,
                               max_new_tokens, num_beams, do_sample, fill_mask):

        model = florence2_model['model']
        processor = florence2_model['processor']

        img = tensor2pil(image[0])
        caption = ""
        results, output_image = process_image(model, processor, img, task, max_new_tokens, num_beams,
                                              do_sample, fill_mask,
                                              text_input)

        if isinstance(results, dict):
            results["width"] = img.width
            results["height"] = img.height

        if output_image == None:
            output_image = image[0].detach().clone().unsqueeze(0)
        else:
            output_image = np.asarray(output_image).astype(np.float32) / 255
            output_image = torch.from_numpy(output_image).unsqueeze(0)

        _, caption = decode_f_bboxes(results)

        return (remove_angle_bracket_content(caption), output_image,)

NODE_CLASS_MAPPINGS = {
    "LayerMask: Florence2Ultra": Florence2Ultra,
    "LayerMask: LoadFlorence2Model": LS_LoadFlorence2Model,
    "LayerUtility: Florence2Image2Prompt": Florence2Image2Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: Florence2Ultra": "LayerMask: Florence2 Ultra(Advance)",
    "LayerMask: LoadFlorence2Model": "LayerMask: Load Florence2 Model(Advance)",
    "LayerUtility: Florence2Image2Prompt": "LayerUtility: Florence2 Image2Prompt(Advance)"
}
