import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import base64
import io
import sys
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from datasets import load_dataset
from PIL import Image
import PIL.Image as PIL_image
import math
from safetensors.torch import load_file
import time


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    res = []
    for i in range(0, len(lst), chunk_size):
        res.append(lst[i : min(i + chunk_size, len(lst))])
    return res


def get_chunk(lst, n, k):
    try:
        chunks = split_list(lst, n)
        return chunks[k]
    except:
        sys.exit()


def eval_model(args):
    # Model
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    model_name = "llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map={"": "cuda"}
    )
    # model_dict = model.state_dict()
    # for i in range(1, 4):
    #     file_path = os.path.join(args.model_path, f"model-0000{i}-of-00003.safetensors")
    #     # 加载 safetensors 文件
    #     state_dict = load_file(file_path)
    #     for name, param in state_dict.items():
    #         if name in model_dict:
    #             # 检查参数大小是否匹配
    #             if model_dict[name].size() == param.size():
    #                 model_dict[name].copy_(param)
    #             else:
    #                 print(f"Skipping {name}, size mismatch: model {model_dict[name].size()} vs state_dict {param.size()}")
    #         else:
    #             print(f"{name} not found in the model")
    # model.load_state_dict(model_dict)

    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    prompt_type = "commonsense_TF-simple_TF-detailed_TF-detailed_TF"
    pt_a, pt_b, pt_c, pt_d = prompt_type.split("-")

    def prompt(statement, prompt_type):
        if prompt_type == "detailed":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Forget real-world common sense and just follow the information provided in the image."
        if prompt_type == "detailed_cot":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Forget real-world common sense and just follow the information provided in the image. Let's think step by step."
        if prompt_type == "detailed_TF":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Forget real-world common sense and just follow the information provided in the image. Only respond in True or False."
        if prompt_type == "detailed_TF_end":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Forget real-world common sense and just follow the information provided in the image. Indicate true or false at the end of your response."
        if prompt_type == "simple":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false?"
        if prompt_type == "simple_cot":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Let's think step by step."
        if prompt_type == "simple_TF":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Only respond in True or False."
        if prompt_type == "commonsense_TF":
            return f"Statement: {statement}\nBased on common sense, is the given statement true or false? Only respond in True or False."
        if prompt_type == "simple_TF_end":
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Indicate true or false at the end of your response."
        if prompt_type == "null":
            return f"Statement: {statement}\nIs the given statement true or false?"
        if prompt_type == "null_TF":
            return f"Statement: {statement}\nIs the given statement true or false? Only respond in True or False."
        raise ValueError("invlaid prompt type!")

    def ctx_prompt(context, statement, prompt_type):
        if prompt_type == "detailed":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Forget real-world common sense and just follow the information provided in the context."
        if prompt_type == "detailed_cot":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Forget real-world common sense and just follow the information provided in the context. Let's think step by step."
        if prompt_type == "detailed_TF":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Forget real-world common sense and just follow the information provided in the context. Only respond in True or False."
        if prompt_type == "detailed_TF_end":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Forget real-world common sense and just follow the information provided in the context. Indicate true or false at the end of your response."
        if prompt_type == "simple":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false?"
        if prompt_type == "simple_cot":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Let's think step by step."
        if prompt_type == "simple_TF":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Only respond in True or False."
        if prompt_type == "simple_TF_end":
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Indicate true or false at the end of your response."
        raise ValueError("invlaid prompt type!")

    def obj_det_statement_template(obj):
        return f"There is {obj} in the given image."

    def infer_true_or_false(response):
        normalized_response = (
            response.lower()
            .replace("\n", " ")
            .replace(",", "")
            .replace(".", "")
            .split(" ")
        )
        for word in normalized_response:
            if word == "true":
                return "True"
            elif word == "false":
                return "False"
        return "NA"

    def get_image_with_most_votes(instance):
        votes = instance["aggregated_human_label_good_images"]
        votes_list = [(a, b) for a, b in votes.items()]
        votes_list_sorted = sorted(votes_list, key=lambda x: x[1], reverse=True)
        return votes_list_sorted[0][0]

    def construct_prompt(prompt):
        if model.config.mm_use_im_start_end:
            prompt = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        return (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        # return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    def get_response(input_ids, image, image_size):
        # start_time = time.time()  # 记录开始时间

        image = process_images([image], image_processor, model.config)[0]
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image.unsqueeze(0).half().cuda(),
                image_sizes=[image_size],
                do_sample=False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=64,
                use_cache=True,
            )

        decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算运行时间
        # print(f"单次推理时间: {elapsed_time:.2f} 秒")  # 打印运行时间

        return decoded_output

    for i, instance in enumerate(tqdm(dataset)):
        good_images = []

        for image_id, vote in instance["aggregated_human_label_good_images"].items():
            if vote >= args.vote_thres:
                good_images.append(image_id)

        if len(good_images) == 0:
            continue

        concept = instance["concept"]
        context = instance["context"]
        factual_context = instance["factual_context"]
        true_statement = instance["true_statement"]
        false_statement = instance["false_statement"]

        input_a1 = construct_prompt(prompt(false_statement, pt_a))  # True
        input_a2 = construct_prompt(prompt(true_statement, pt_a))  # False

        input_b1 = construct_prompt(
            prompt(obj_det_statement_template(instance["existent_noun"]), pt_b)
        )  # True
        input_b2 = construct_prompt(
            prompt(obj_det_statement_template(instance["non-existent_noun"]), pt_b)
        )  # False

        input_c1 = construct_prompt(ctx_prompt(context, true_statement, pt_c))  # True
        input_c2 = construct_prompt(ctx_prompt(context, false_statement, pt_c))  # False

        input_d1 = construct_prompt(prompt(true_statement, pt_d))  # True
        input_d2 = construct_prompt(prompt(false_statement, pt_d))  # False
        instance["prompt"] = prompt(true_statement, pt_d)

        image_dir = (
            args.image_dir
            + f"/{instance['concept']}/{instance['context_id']}_{instance['context']}"
        )
        factual_image_dir = (
            args.factual_image_dir
            + f"/{instance['concept']}/{instance['context_id']}_{instance['factual_context']}"
        )

        best_cf_image_id = get_image_with_most_votes(instance)
        instance["bc_input_image_id"] = best_cf_image_id
        counterfactual_image_path = image_dir + f"/{best_cf_image_id}.jpg"
        counterfactual_image = Image.open(counterfactual_image_path).convert("RGB")
        factual_image_path = factual_image_dir + "/0.jpg"
        factual_image = Image.open(factual_image_path).convert("RGB")

        instance[model_name + "_a1_image"] = {
            "response": get_response(input_a1, factual_image, factual_image.size)
        }
        instance[model_name + "_a2_image"] = {
            "response": get_response(input_a2, factual_image, factual_image.size)
        }
        instance[model_name + "_a1_image"]["answer"] = infer_true_or_false(
            instance[model_name + "_a1_image"]["response"]
        )
        instance[model_name + "_a2_image"]["answer"] = infer_true_or_false(
            instance[model_name + "_a2_image"]["response"]
        )

        instance[model_name + "_b1"] = {
            "response": get_response(
                input_b1, counterfactual_image, counterfactual_image.size
            )
        }
        instance[model_name + "_b2"] = {
            "response": get_response(
                input_b2, counterfactual_image, counterfactual_image.size
            )
        }
        instance[model_name + "_b1"]["answer"] = infer_true_or_false(
            instance[model_name + "_b1"]["response"]
        )
        instance[model_name + "_b2"]["answer"] = infer_true_or_false(
            instance[model_name + "_b2"]["response"]
        )

        instance[model_name + "_c1_image"] = {
            "response": get_response(
                input_c1, counterfactual_image, counterfactual_image.size
            )
        }
        instance[model_name + "_c2_image"] = {
            "response": get_response(
                input_c2, counterfactual_image, counterfactual_image.size
            )
        }
        instance[model_name + "_c1_image"]["answer"] = infer_true_or_false(
            instance[model_name + "_c1_image"]["response"]
        )
        instance[model_name + "_c2_image"]["answer"] = infer_true_or_false(
            instance[model_name + "_c2_image"]["response"]
        )

        instance[model_name + "_d1"] = {}
        instance[model_name + "_d2"] = {}
        for good_image_id in good_images:
            image_path = image_dir + f"/{good_image_id}.jpg"
            image = Image.open(image_path).convert("RGB")

            instance[model_name + "_d1"][good_image_id] = {
                "response": get_response(input_d1, image, image.size)
            }
            instance[model_name + "_d2"][good_image_id] = {
                "response": get_response(input_d2, image, image.size)
            }
            instance[model_name + "_d1"][good_image_id]["answer"] = infer_true_or_false(
                instance[model_name + "_d1"][good_image_id]["response"]
            )
            instance[model_name + "_d2"][good_image_id]["answer"] = infer_true_or_false(
                instance[model_name + "_d2"][good_image_id]["response"]
            )

    with open(args.answers_file, "w") as f:
        json.dump(dataset, f)
    print(f"Chunk {args.chunk_idx} finished!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dp",
        "--data_path",
        dest="data_path",
        type=str,
        action="store",
        default="./datasets/VLind-Bench/VLind-Bench Dataset/data.json",
    )
    parser.add_argument(
        "-af", "--answers-file", dest="answers_file", type=str, action="store"
    )
    parser.add_argument(
        "-id",
        "--image_dir",
        dest="image_dir",
        type=str,
        action="store",
        default="./datasets/VLind-Bench/VLind-Bench Dataset/images/counterfactual",
    )
    parser.add_argument(
        "-fid",
        "--factual_image_dir",
        dest="factual_image_dir",
        type=str,
        action="store",
        default="./datasets/VLind-Bench/VLind-Bench Dataset/images/factual",
    )
    parser.add_argument(
        "-vt", "--vote_thres", dest="vote_thres", type=int, action="store", default=2
    )

    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    print(args.conv_mode)

    eval_model(args)
