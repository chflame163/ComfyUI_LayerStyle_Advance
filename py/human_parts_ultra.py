# layerstyle advance

import os
from typing import Tuple
import torch
import numpy as np
from PIL import Image, ImageEnhance
import folder_paths
from .imagefunc import pil2tensor, tensor2pil, image2mask, mask2image, log, RGB2RGBA, histogram_remap
from .imagefunc import generate_VITMatte_trimap, generate_VITMatte, mask_edge_detail, guided_filter_alpha

models_dir_path = os.path.join(folder_paths.models_dir, "onnx", "human-parts")
model_url = "https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir_path, "deeplabv3p-resnet50-human.onnx")


class LS_HumanPartsUltra:
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.

    """

    def __init__(self):
        self.NODE_NAME = 'HumanPartsUltra'

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "human_parts_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'vitmatte-base-composition-1k', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "face": ("BOOLEAN", {"default": False, "label_on": "enabled(脸)", "label_off": "disabled(脸)"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled(头发)", "label_off": "disabled(头发)"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "enabled(眼镜)", "label_off": "disabled(眼镜)"}),
                "top_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(上装)", "label_off": "disabled(上装)"}),
                "bottom_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(下装)", "label_off": "disabled(下装)"}),
                "torso_skin": ("BOOLEAN", {"default": False, "label_on": "enabled(躯干)", "label_off": "disabled(躯干)"}),
                "left_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(左臂)", "label_off": "disabled(左臂)"}),
                "right_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(右臂)", "label_off": "disabled(右臂)"}),
                "left_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(左腿)", "label_off": "disabled(左腿)"}),
                "right_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(右腿)", "label_off": "disabled(右腿)"}),
                "left_foot": ("BOOLEAN", {"default": False, "label_on": "enabled(左脚)", "label_off": "disabled(左脚)"}),
                "right_foot": ("BOOLEAN", {"default": False, "label_on": "enabled(右脚)", "label_off": "disabled(右脚)"}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 8, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": (
                "FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": (
                "FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            }
        }

    def human_parts_ultra(self, image, face, hair, glasses, top_clothes, bottom_clothes,
                          torso_skin, left_arm, right_arm, left_leg, right_leg, left_foot, right_foot,
                          detail_method, detail_erode, detail_dilate, black_point, white_point,
                          process_detail, device, max_megapixels):
        """
        Return a Tensor with the mask of the human parts in the image.
        """
        import onnxruntime as ort

        model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider', 'TensorrtExecutionProvider'])
        ret_images = []
        ret_masks = []
        for img in image:
            orig_image = tensor2pil(img).convert('RGB')

            human_parts_mask, _ = self.get_mask(orig_image, model=model, rotation=0, background=False,
                                          face=face, hair=hair, glasses=glasses,
                                          top_clothes=top_clothes, bottom_clothes=bottom_clothes,
                                          torso_skin=torso_skin, left_arm=left_arm, right_arm=right_arm,
                                          left_leg=left_leg, right_leg=right_leg,
                                          left_foot=right_foot, right_foot=right_foot)
            _mask = tensor2pil(human_parts_mask).convert('L')
            brightness_image = ImageEnhance.Brightness(_mask)
            _mask = brightness_image.enhance(factor=1.08)
            _mask = image2mask(_mask)

            if detail_method == 'VITMatte(local)':
                local_files_only = True
            else:
                local_files_only = False
            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(img.unsqueeze(0), _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(img.unsqueeze(0), _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                              max_megapixels=max_megapixels, method=detail_method)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

    def get_mask(self, pil_image:Image, model, rotation:float, **kwargs) -> tuple:
        """
        Return a Tensor with the mask of the human parts in the image.

        The rotation parameter is not used for now. The idea is to propose rotation to help
        the model to detect the human parts in the image if the character is not in a casual position.
        Several tests have been done, but the model seems to fail to detect the human parts in these cases,
        and the rotation does not help.
        """

        # classes used in the model
        classes = {
            "background": 0,
            "hair": 2,
            "glasses": 4,
            "top_clothes": 5,
            "bottom_clothes": 9,
            "torso_skin": 10,
            "face": 13,
            "left_arm": 14,
            "right_arm": 15,
            "left_leg": 16,
            "right_leg": 17,
            "left_foot": 18,
            "right_foot": 19,
        }

        original_size = pil_image.size  # to resize the mask later
        # resize to 512x512 as the model expects
        pil_image = pil_image.resize((512, 512))
        center = (256, 256)

        if rotation != 0:
            pil_image = pil_image.rotate(rotation, center=center)

        # normalize the image
        image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1
        image_np = np.expand_dims(image_np, axis=0)

        # use the onnx model to get the mask
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: image_np})
        result = np.array(result[0]).argmax(axis=3).squeeze(0)

        score: int = 0

        mask = np.zeros_like(result)
        for class_name, enabled in kwargs.items():
            if enabled and class_name in classes:
                class_index = classes[class_name]
                detected = result == class_index
                mask[detected] = 255
                score += mask.sum()

        # back to the original size
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        if rotation != 0:
            mask_image = mask_image.rotate(-rotation, center=center)

        mask_image = mask_image.resize(original_size)

        # and back to numpy...
        mask = np.array(mask_image).astype(np.float32) / 255

        # add 2 dimensions to match the expected output
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        # ensure to return a "binary mask_image"

        del image_np, result  # free up memory, maybe not necessary
        return (torch.from_numpy(mask.astype(np.uint8)), score)


NODE_CLASS_MAPPINGS = {
    "LayerMask: HumanPartsUltra": LS_HumanPartsUltra
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: HumanPartsUltra": "LayerMask: Human Parts Ultra(Advance)"
}
