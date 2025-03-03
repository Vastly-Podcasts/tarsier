from dataset.utils import load_model_and_processor
from dataset.utils import *
import os
from tqdm import tqdm

def process_batch(model, processor, prompts, video_files, generate_kwargs):
    """
    model, processor = load_model_and_processor("omni-research/Tarsier-7b", 8)

    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 512,
        "top_p": 1,
        "temperature": 0,
        "use_cache": True
    }

    video_files = [r"/root/tarsier/assets/videos/demo_cli_example.mp4", r"/root/tarsier/assets/videos/coffee.gif"]
    prompts = ["<video>\nDescribe the video in detail.", "<video>\nDescribe the video in detail."]
    response = process_batch(model, processor, prompts, video_files, generate_kwargs)
    """
    import torch
    inputs_list = [processor(prompt, video_file, edit_prompt=True) for prompt, video_file in zip(prompts, video_files)]
    input_ids_list = [inputs['input_ids'] for inputs in inputs_list]
    pixel_values_list = [inputs['pixel_values'] for inputs in inputs_list]
    inputs = {
        'input_ids': torch.concat(input_ids_list, dim=0).to(model.device),
        'pixel_values': torch.concat(pixel_values_list, dim=0).to(model.device)
    }
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    return [processor.tokenizer.decode(output[inputs['input_ids'][index].shape[0]:], skip_special_tokens=True) for index, output in enumerate(outputs)]
    
def process_one(model, processor, prompt, video_file, generate_kwargs):
    inputs = processor(prompt, video_file, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        print(f"Prompt: {inputs.pop('prompt')}")
    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text

def run():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--instruction', type=str, default="", help='Input prompt.')
    parser.add_argument('--input_path', type=str, default="assets/examples", help='Path to video/image; or Dir to videos/images')
    parser.add_argument("--max_n_frames", type=int, default=8, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")

    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_name_or_path, max_n_frames=args.max_n_frames)
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True
    }
    assert os.path.exists(args.input_path), f"input_path not exist: {args.input_path}"
    if os.path.isdir(args.input_path):
        input_files = [os.path.join(args.input_path, fn) for fn in os.listdir(args.input_path) if get_visual_type(fn) in ['video', 'gif', 'image']]
    elif get_visual_type(args.input_path) in ['video', 'gif', 'image']:
        input_files = [args.input_path]
    assert len(input_files) > 0, f"None valid input file in: {args.input_path} {VALID_DATA_FORMAT_STRING}"

    for input_file in tqdm(input_files, desc="Generating..."):
        visual_type = get_visual_type(input_file)
        if args.instruction:
            prompt = args.instruction
            prompt = "<video>\n" + prompt.replace("<image>", "").replace("<video>", "")
        else:
            if visual_type == 'image':
                prompt = "<image>\nDescribe the image in detail."
            else:
                prompt = "<video>\nDescribe the video in detail."
        
        pred = process_one(model, processor, prompt, input_file, generate_kwargs)
        print(f"Prediction: {pred}")
        print('-'*100)

if __name__ == "__main__":
    run() 