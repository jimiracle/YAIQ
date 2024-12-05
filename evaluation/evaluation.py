import argparse
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import os
import pandas as pd
import json


# Argparse 추가
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation args parser")
    parser.add_argument("--mode", type=str, default="default", help="Mode of operation")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache model/processor directory")
    parser.add_argument("--original_model", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="original model path after finetuning")
    parser.add_argument("--dataset_path", type=str, default="eduardtoni/MENSA-visual-iq-test", help="Path to the dataset")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, default="results_onevision.json", help="Path to save the output JSON")
    return parser.parse_args()


# Main 함수 추가
def main():
    args = parse_args()

    # Load the model
    if args.mode = "peft":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", 
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.cache_dir, 
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True, 
        )
        processor = AutoProcessor.from_pretrained(args.original_model)
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            device_map="auto",
            cache_dir=args.cache_dir
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=args.cache_dir)

    # Load prompt JSON file
    with open(args.prompt_path, "r") as file:
        data = json.load(file)

    conversation_1 = data["conversation_1"]
    prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
    prompts = [prompt_1]

    dataset = load_dataset(args.dataset_path)
    dataset = dataset['train']

    results = []
    for data in dataset:
        choice_image = data['choices_images']
        question_image = data['question_img']
        choice_image = [Image.open(os.path.join(args.image_path, f)) for f in sorted(data['choices_images'])]
        path = []
        path.append(question_image)
        image = path + choice_image

        inputs = processor(images=image, text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=500)
        result_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({
            "text_prompt": prompts,
            "output": result_text,
            "answer": data['correct_answer'],
        })
        print(results[-1])


    # Save the results list to a JSON file
    with open(args.output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)  # Use indent=4 for pretty printing

    print(f"Results saved to {args.output_path}")



if __name__ == "__main__":
    main()
