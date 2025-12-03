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
    
    # Patch ModelOutput when transformers.utils is imported
    if name == 'transformers.utils' or name.startswith('transformers.utils.'):
        try:
            if hasattr(module, 'ModelOutput'):
                original_init_subclass = module.ModelOutput.__init_subclass__
                @classmethod  
                def safe_init_subclass(cls, **kw):
                    try:
                        return original_init_subclass.__func__(cls, **kw)
                    except ValueError:
                        pass  # Skip docstring errors
                module.ModelOutput.__init_subclass__ = safe_init_subclass
        except:
            pass
    
    return module

builtins.__import__ = _patched_import

import io
import torch
from unittest.mock import patch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys
from transformers.dynamic_module_utils import get_imports
from transformers import PreTrainedModel
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
    """
    modeling_file = os.path.join(model_path, "modeling_florence2.py")
    
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
    
    # CRITICAL FIX: Replace ModelOutput inheritance to avoid docstring validation
    import re
    
    # Check if file uses ModelOutput
    if 'ModelOutput' in content and 'from dataclasses import dataclass' not in content:
        # Add dataclass import at the top (after existing imports)
        import_section = """# PATCHED: Simple output class to bypass ModelOutput docstring validation
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import torch

@dataclass 
class Florence2BaseModelOutput:
    \"\"\"
    Base output class for Florence2 models.
    
    Args:
        loss: Optional loss tensor.
        logits: Model logits.
        last_hidden_state: Last hidden state.
        hidden_states: All hidden states.
        attentions: Attention weights.
    \"\"\"
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    encoder_attentions: Optional[Tuple[torch.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    decoder_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    image_hidden_states: Optional[torch.Tensor] = None

"""
        # Insert after the first import block
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif insert_idx > 0 and not line.strip().startswith('#') and line.strip() and not line.startswith('import') and not line.startswith('from'):
                break
        
        lines.insert(insert_idx, import_section)
        content = '\n'.join(lines)
        
        # Replace all *Output(ModelOutput) with *Output(Florence2BaseModelOutput)
        content = re.sub(
            r'class (\w+Output)\(ModelOutput\)',
            r'class \1(Florence2BaseModelOutput)',
            content
        )
        log("[PATCH] Replaced ModelOutput with Florence2BaseModelOutput")
    
    # Write the patched file
    with open(modeling_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log(f"[PATCH] File patched successfully!")
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
    
    # Suppress docstring validation errors by patching ModelOutput
    try:
        from transformers.utils.generic import ModelOutput
        if hasattr(ModelOutput, '__init_subclass__'):
            original_init_subclass = ModelOutput.__init_subclass__
            @classmethod
            def safe_init_subclass(cls, **kwargs):
                try:
                    return original_init_subclass.__func__(cls, **kwargs)
                except ValueError as e:
                    if "Args" in str(e) or "Parameters" in str(e):
                        log(f"[LOAD] Suppressed docstring error for {cls.__name__}")
                        return None
                    raise
            ModelOutput.__init_subclass__ = safe_init_subclass
            log("[LOAD] Patched ModelOutput.__init_subclass__")
    except Exception as e:
        log(f"[LOAD] Could not patch ModelOutput: {e}")
    
    # Load the model
    try:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            log(f"[LOAD] Model loaded: {type(model).__name__}")
            
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            log(f"[LOAD] Processor loaded: {type(processor).__name__}")
    except Exception as e:
        log(f"[LOAD] Error: {str(e)}", message_type='error')
        return (None, None)
    
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
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        do_sample=do_sample,
        num_beams=num_beams,
    )
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
