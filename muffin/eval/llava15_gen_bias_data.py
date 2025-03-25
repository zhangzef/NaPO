import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import base64
import io
import sys
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from datasets import load_dataset
from PIL import Image
import PIL.Image as PIL_image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    res = []
    for i in range(0, len(lst), chunk_size):
        res.append(lst.select(range(i, min(i+chunk_size, len(lst)))))
    return res

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = PIL_image.open(img_io).convert('RGB')
    return image

def add_new_column(example, idx):
    example['new_column'] = new_column_data[idx]
    return example

def eval_model(args):
    # Model
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    model_name = 'llava-v1.5-7b'
    white_image = Image.open('./examples/white_image.png').convert('RGB')
    black_image = Image.open('./examples/black_image.png').convert('RGB')
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map={"": 'cuda'})

    dataset = get_chunk(load_dataset(args.data_path, split='train'), args.num_chunks, args.chunk_idx)
    qs_only_answers = []
    white_qs_only_answers = []
    black_qs_only_answers = []
    image_only_answers = []
    for line in tqdm(dataset):
        qs = line["question"]
        image = bytes_to_PIL_image(line['image']['bytes'])

        # construct question only prompt
        conv_qs_only = conv_templates[args.conv_mode].copy()
        conv_qs_only.append_message(conv_qs_only.roles[0], qs)
        conv_qs_only.append_message(conv_qs_only.roles[1], None)
        prompt_qs_only = conv_qs_only.get_prompt()
        input_ids_qs_only = tokenizer_image_token(prompt_qs_only, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # construct image only prompt
        if model.config.mm_use_im_start_end:
            image_only_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
        else:
            image_only_qs = DEFAULT_IMAGE_TOKEN + '\n'
        conv_image_only = conv_templates[args.conv_mode].copy()
        conv_image_only.append_message(conv_image_only.roles[0], image_only_qs)
        conv_image_only.append_message(conv_image_only.roles[1], None)
        prompt_image_only = conv_image_only.get_prompt()
        input_ids_image_only = tokenizer_image_token(prompt_image_only, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        
        input_ids_bias = torch.stack([input_ids_image_only, input_ids_image_only, input_ids_image_only], dim=0)
        image_tensor = process_images([image, white_image, black_image], image_processor, model.config)

        with torch.inference_mode():
            output_ids_qs_only = model.generate(
                input_ids_qs_only,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
            output_ids_bias = model.generate(
                input_ids_bias,
                images=image_tensor.half().cuda(),
                image_sizes=[image.size, white_image.size, black_image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs_qs_only = tokenizer.batch_decode(output_ids_qs_only, skip_special_tokens=True)[0].strip()
        outputs_bias = tokenizer.batch_decode(output_ids_bias, skip_special_tokens=True)
        outputs_bias = [outputs_bias[i].strip() for i in range(len(outputs_bias)) if len(outputs_bias[i]) > 0]

        qs_only_answers.append(outputs_qs_only)
        image_only_answers.append(outputs_bias[0])
        white_qs_only_answers.append(outputs_bias[1])
        black_qs_only_answers.append(outputs_bias[2])

    def add_new_columns(example, idx):
        example['question_only_None'] = qs_only_answers[idx]
        example['image_only'] = image_only_answers[idx]
        example['question_only_white'] = white_qs_only_answers[idx]
        example['question_only_black'] = black_qs_only_answers[idx]
        return example
    modified_dataset = dataset.map(add_new_columns, with_indices=True)
    modified_dataset.to_parquet(os.path.join(args.output_path, f'chunk_{args.chunk_idx:03}.parquet'))
    print(f'Chunk {args.chunk_idx} finished!!!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default='./datasets/RLAIF-V-Dataset')
    parser.add_argument("--output-path", type=str, default='./datasets/RLAIF-V-Bias-Dataset')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    print(args.conv_mode)

    eval_model(args)
