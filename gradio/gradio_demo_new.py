import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io

# Add training source path
sys.path.append("/root/zhuquanhao/CoVT/train/src")

from transformers import AutoProcessor
from training.covt_qwen2_5_vl import CoVTForConditionalGeneration
from training.constants import SAM_PAD_TOKEN

# ================= Configuration Area =================
# DEFAULT_MODEL_NAME = "/root/output/lora_merged/lora_stage1_merged"
DEFAULT_MODEL_NAME = "/root/output/lora_merged/lora_stage234_merged"
DEFAULT_SAM_CHECKPOINT = "/root/zhuquanhao/CoVT/train/src/anchors/segment_anything/ckpt/medsam_vit_b.pth"
# ======================================================

# Global cache
_cached_model = None
_cached_processor = None

def load_model_and_processor(model_name: str):
    print(f"Loading model from: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CoVTForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Initialize Anchor Models (SAM)
    # The list needs to include 'sam' to initialize SAM related components
    # We might need to ensure the checkpoint path is correct inside the model config or pass it if possible
    # In inference_demo.py, it calls model.get_anchor_model_ids(['sam'])
    # The actual loading of SAM weights might happen inside get_anchor_model_ids or rely on default paths.
    # Looking at inference_demo.py, sam_checkpoint variable is defined but not explicitly passed to get_anchor_model_ids?
    # Ah, let's assume get_anchor_model_ids loads defaults or we need to configure it.
    # inference_demo.py just calls: model.get_anchor_model_ids(['sam'])
    model.get_anchor_model_ids(['sam'])
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )
    
    # Setup SAM token
    if SAM_PAD_TOKEN not in processor.tokenizer.get_vocab():
        print(f"Warning: {SAM_PAD_TOKEN} not found in tokenizer vocab. Adding it.")
        processor.tokenizer.add_tokens([SAM_PAD_TOKEN], special_tokens=True)
    
    sam_token_id = processor.tokenizer.convert_tokens_to_ids(SAM_PAD_TOKEN)
    model.sam_token_idx = sam_token_id
    
    return model, processor

def get_cached_model_and_processor(model_name: str = DEFAULT_MODEL_NAME):
    global _cached_model, _cached_processor
    if _cached_model is not None and _cached_processor is not None:
        return _cached_model, _cached_processor
    
    _cached_model, _cached_processor = load_model_and_processor(model_name)
    return _cached_model, _cached_processor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def run_inference(
    image,
    question,
    max_new_tokens,
    temperature,
    top_p,
):
    if image is None:
        return "Please upload an image.", None, 0.0

    model, processor = get_cached_model_and_processor()
    device = model.device

    # Prepare Image
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
        image_ref = image
    else:
        pil_image = image.convert("RGB")
        image_ref = "gradio_image"

    # Prepare Inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt"
    )
    
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    
    start_time = time.time()
    
    # Generate
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(temperature > 0.0),
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Decode Text
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=False) # Keep special tokens to see if SAM tokens exist
    
    # --- Segmentation Logic ---
    output_image = pil_image.copy()
    
    # Find SAM tokens
    sam_token_id = model.sam_token_idx
    full_input_ids = generated_ids[0].unsqueeze(0)
    sam_token_mask = (full_input_ids == sam_token_id)
    
    if sam_token_mask.any():
        print(f"Found {sam_token_mask.sum()} SAM tokens.")
        
        # Forward pass for hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids,
                pixel_values=inputs['pixel_values'],
                image_grid_thw=inputs['image_grid_thw'],
                output_hidden_states=True
            )
        
        last_hidden_state = outputs.hidden_states[-1]
        sam_hidden_features = last_hidden_state[sam_token_mask].unsqueeze(0)
        
        if hasattr(model, 'apply_rope_custome'):
            sam_hidden_features = model.apply_rope_custome(sam_hidden_features)
            
        # Decode Masks
        sam_token_embeddings = model.sam_projection(sam_hidden_features)
        sam_query = model.sam_query_vectors.unsqueeze(0)
        sam_proj = F.normalize(sam_token_embeddings, dim=-1)
        
        sam_attn_output, _ = model.sam_cross_attention(
            query=sam_query,
            key=sam_proj,
            value=sam_proj
        )
        
        if hasattr(model.anchor_models, 'sam'):
            model.anchor_models.sam.to(device)
            
        sam_embed = model.anchor_models.get_sam_embed(pil_image)
        token_embeddings = sam_attn_output[0]
        
        pred_masks = model.anchor_models.decode_sam_embed_with_tokens(
            sam_embed,
            pil_image,
            token_embeddings
        )
        
        # Visualization
        original_w, original_h = pil_image.size
        
        # Resize masks if needed
        pred_masks_resized = F.interpolate(
            pred_masks.unsqueeze(1), 
            size=(original_h, original_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        masks_np = pred_masks_resized.detach().cpu().numpy()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(pil_image)
        
        for i in range(masks_np.shape[0]):
            mask = masks_np[i] > 0
            if mask.sum() > 0:
                show_mask(mask, ax, random_color=True)
        
        ax.axis('off')
        
        # Save plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        output_image = Image.open(buf).convert("RGB")
        plt.close(fig)
        
    # Clean up text output (remove special tokens for display if desired, or keep them)
    # Usually for user display we might want to remove <|endoftext|> etc, but keep <think> tags?
    # Let's clean it up slightly but maybe keep structure.
    answer_clean = answer.replace(SAM_PAD_TOKEN, "[SEG]") 
    
    return answer_clean, output_image, elapsed

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# MedVCoT Gradio Demo\n"
            "Upload an image and input a question. If the model decides to segment, masks will be shown."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input Image", type="pil")
                question_input = gr.Textbox(label="Question", lines=2)
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_new_tokens = gr.Slider(label="Max New Tokens", minimum=1, maximum=1024, value=512, step=1)
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                
                # Example
                gr.Markdown("### Example")
                example_image_path = "/root/datasets/all_images_test/pvqa_test_0450.jpg"
                if os.path.exists(example_image_path):
                    example_image = Image.open(example_image_path).convert("RGB")
                    gr.Examples(
                        examples=[
                            [
                                example_image,
                                "Is endocrine present?",
                                512,
                                0.0,
                                0.9
                            ]
                        ],
                        inputs=[image_input, question_input, max_new_tokens, temperature, top_p],
                        examples_per_page=1
                    )
                else:
                    gr.Markdown(f"Example image not found at {example_image_path}")
                
                run_button = gr.Button("Run Inference", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(label="Answer", lines=10)
                image_output = gr.Image(label="Segmentation Result", type="pil")
                elapsed_output = gr.Number(label="Elapsed time (seconds)")

        run_button.click(
            fn=run_inference,
            inputs=[image_input, question_input, max_new_tokens, temperature, top_p],
            outputs=[answer_output, image_output, elapsed_output]
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
