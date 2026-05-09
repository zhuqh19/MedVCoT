import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gradio as gr
import os

# ================= Configuration Area =================
# You can change these defaults as you like
DEFAULT_MODEL_NAME = "/root/CoVT-7B-seg"
DEFAULT_CKPT_PATH = None  # Or set to your local checkpoint path
# ======================================================

# Global cache for model and processor to avoid re-loading every call
_cached_model = None
_cached_processor = None


def load_model_and_processor(
    model_name: str,
    ckpt: str = None,
):
    """
    Load a single CoVT-7B model and its corresponding processor.
    """
    if ckpt is not None:
        print(f"Loading model from ckpt: {ckpt}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(
            ckpt,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )
    else:
        print(f"Loading model from hub: {model_name}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )

    return model, processor


def get_cached_model_and_processor(
    model_name: str = DEFAULT_MODEL_NAME,
    ckpt: str = DEFAULT_CKPT_PATH,
):
    """
    Lazy-load and cache the model and processor so they are not reloaded every request.
    """
    global _cached_model, _cached_processor

    # If already loaded, just return them
    if _cached_model is not None and _cached_processor is not None:
        return _cached_model, _cached_processor

    # Otherwise load and cache
    _cached_model, _cached_processor = load_model_and_processor(
        model_name=model_name,
        ckpt=ckpt,
    )
    return _cached_model, _cached_processor


def run_single_inference(
    model,
    processor,
    image,  # can be either a PIL.Image or a path string
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    """
    Single inference: given one image and one question, return answer and elapsed time.
    """
    # 1) Prepare conversation
    # For Gradio we usually get a PIL image, but we also support a path string for compatibility.
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
        image_ref = image  # path for the "image" field
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        # When using PIL image in chat template, you can pass a placeholder
        # and rely on 'images' argument in processor; here we still need a "dummy" reference.
        image_ref = "gradio_image"  # this is not used as a real path, just a placeholder
    else:
        raise ValueError("image must be a PIL.Image or a path string.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": question},
            ],
        }
    ]

    # 2) Apply chat template
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3) Encode image and text
    inputs = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt"
    )

    # Move inputs to the same device as the model
    device = model.device
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    # 4) Timing + generation
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start

    # 5) Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=True)

    return answer, elapsed


def gradio_inference(
    image,
    question,
    max_new_tokens,
    temperature,
    top_p,
):
    """
    Wrapper function for Gradio that calls the inference logic and returns answer + time cost.
    """
    if image is None:
        return "Please upload an image.", 0.0

    # Get (or load) model and processor
    model, processor = get_cached_model_and_processor()

    # Run inference
    answer, elapsed = run_single_inference(
        model=model,
        processor=processor,
        image=image,  # filepath string from Gradio
        question=question,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=(temperature > 0.0),
    )

    return answer, elapsed


# ===================== Gradio UI =====================

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# CoVT-7B Gradio Demo\n"
            "Upload an image and input a question to run visual question answering."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Input Image",
                    type="pil"  # 使用 PIL，避免文件路径搬运限制
                )
                question_input = gr.Textbox(
                    label="Question",
                    value="",
                    lines=2
                )
                max_new_tokens = gr.Slider(
                    label="max_new_tokens",
                    minimum=1,
                    maximum=1024,
                    value=512,
                    step=1
                )
                temperature = gr.Slider(
                    label="temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01
                )
                top_p = gr.Slider(
                    label="top_p",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.01
                )

                # ----- Examples (点击后填充到输入组件) -----
                gr.Markdown("### Example")
                example_image_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "assets", "clouds.png")
                )
                example_image = Image.open(example_image_path).convert("RGB")
                gr.Examples(
                    examples=[
                        [
                            example_image,
                            "Describe the scene in the picture in detail, and find out how many clouds are in the sky. Use segmentation, depth map, and perception feature information of the image to answer this question."
                        ]
                    ],
                    inputs=[image_input, question_input],
                    examples_per_page=1
                )
                # -----------------------------------------

                run_button = gr.Button("Run Inference")

            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10
                )
                elapsed_output = gr.Number(
                    label="Elapsed time (seconds)"
                )

        run_button.click(
            fn=gradio_inference,
            inputs=[image_input, question_input, max_new_tokens, temperature, top_p],
            outputs=[answer_output, elapsed_output]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # You can set share=True if you want a public link
    demo.launch()