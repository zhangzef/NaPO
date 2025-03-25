import os
from pprint import pprint
from llava.model import *
import torch
import random
import copy
import transformers
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn import Module
import PIL.Image as PIL_image
from functools import partial
import io
import json
import os.path as op
import torch.utils.data as torch_data
import tokenizers
import datasets as hf_datasets
from packaging import version
import itertools
import tqdm
from torchvision.transforms import v2
from muffin.train.train_utils import preprocess_v1, SFT_collator_fn, expand_image_token
from muffin.eval.muffin_inference_logp import (
    InferenceSampler,
    get_batch_logps,
    write_logp_to_preference_parquet,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed
import torch.nn.functional as F
from muffin.train.trainers import (
    ZephyrTrainer,
    dpo_loss,
    dpo_lq_loss,
)
import numpy as np


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)
IMAGE_TOKEN_INDEX = -200  # from llava 1.5, used to determin image in forward function
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

keys_list = ["win", "rej", "image_only", "question_only"]
# keys_list = ["win", "rej"]


def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = PIL_image.open(img_io).convert("RGB")
    return image


def crop_images(images):
    new_images = []
    if len(images) == 0:
        return []
    if torch.is_tensor(images[0]):
        for image in images:
            resize_cropper = v2.RandomResizedCrop(
                size=image.size()[-2:], scale=(0.01, 0.2)
            )
            image = resize_cropper(image).unsqueeze(0)
            new_images.append(image)
        return torch.cat(new_images, dim=0)
    else:
        for image in images:
            resize_cropper = v2.RandomResizedCrop(
                size=image.size[::-1], scale=(0.01, 0.2)
            )
            image = resize_cropper(image)
            new_images.append(image)
        return new_images


#################################################################################################
# BiasDataset


class RLAIFVBiasDataset(torch_data.Dataset):
    def __init__(
        self,
        data_dir: str,
        reference_model=None,
        tokenizer=None,
        image_token_len=None,
        img_processor=None,
        use_im_start_end=True,
        is_llava15=False,
    ):
        super().__init__()
        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        cache_dir = data_dir + "-7b-logp"

        if not op.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        data_path = [
            file
            for file in os.listdir(cache_dir)
            if file.endswith(".parquet") and "logp" in file
        ]
        self.data_path = data_dir

        if len(data_path) == 0:
            assert (
                reference_model is not None
            ), "`reference_model` is mandatory when logps do not exist."

            # hf_data = (
            #     hf_datasets.load_dataset(data_dir)["train"]
            #     .select(range(20))
            #     .cast_column("image", hf_datasets.Image(decode=False))
            # )
            hf_data = hf_datasets.load_dataset(data_dir)["train"].cast_column(
                "image", hf_datasets.Image(decode=False)
            )
            print(f"ingerence logp samples: {len(hf_data)}")
            inference_logp(
                reference_model,
                tokenizer,
                hf_data,
                cache_dir,
                image_token_len,
                img_processor,
                use_im_start_end,
                is_llava15=is_llava15,
            )

            torch.distributed.barrier()

            self.data = hf_datasets.load_dataset(cache_dir)["train"].cast_column(
                "image", hf_datasets.Image(decode=False)
            )
        else:
            self.data = hf_datasets.load_dataset(cache_dir)["train"].cast_column(
                "image", hf_datasets.Image(decode=False)
            )

        self.line_idx = list(range(len(self.data)))
        random.shuffle(self.line_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 从数据集中根据索引获取样本
        sample = self.data[self.line_idx[index]]

        # 构建问题字典，包括来源和问题的文本内容
        question = {"from": "human", "value": f"<image>\n{sample['question']}"}
        win = {"from": "gpt", "value": sample["chosen"]}
        rej = {"from": "gpt", "value": sample["rejected"]}
        image_only = {"from": "gpt", "value": sample["image_only"]}
        question_only = {"from": "gpt", "value": sample["question_only"]}

        # 将样本中的图像字节转换为PIL图像
        image = bytes_to_PIL_image(sample["image"]["bytes"])

        # 构建样本的元信息字典
        metainfo = {
            "origin_dataset": sample["origin_dataset"],  # 原始数据集
            "origin_split": sample["origin_split"],  # 原始分割
            "origin_idx": sample["idx"],  # 原始索引
            "image_id": sample["image_path"],  # 图像ID或路径
        }

        # 构造返回的数据字典
        data_dict = {
            "image": image,  # PIL图像
            "question": question,  # 问题字典
            "win": win,  # GPT选择的答案字典
            "rej": rej,  # GPT拒绝的答案字典
            "image_only": image_only,
            "question_only": question_only,
            "idx": sample["idx"],  # 样本索引
            "metainfo": metainfo,  # 元信息字典
        }

        # 加载并解析样本中的logps（对数概率）信息
        logps = json.loads(sample["logps"])

        # 5
        # 根据logps的类型（列表或字典），解析出相应的对数概率信息并添加到data_dict中
        if type(logps) == type([]):
            for i in range(0, len(logps), 3):
                print(i / 3)
                data_dict[f"ref_{keys_list[int(i/3)]}_logp"] = logps[i]
                data_dict[f"ref_{keys_list[int(i/3)]}_avg_logp"] = logps[i + 1]
                data_dict[f"ref_{keys_list[int(i/3)]}_per_token_logp"] = logps[i + 2]
        else:
            for i in range(0, len(logps["logps"]), 3):
                data_dict[f"ref_{keys_list[int(i/3)]}_logp"] = logps["logps"][i]
                data_dict[f"ref_{keys_list[int(i/3)]}_avg_logp"] = logps["logps"][i + 1]
                data_dict[f"ref_{keys_list[int(i/3)]}_per_token_logp"] = logps["logps"][
                    i + 2
                ]
        # 返回构建好的数据字典
        return data_dict


def encode_multimodal_preference_sample(
    source, tokenizer, multimodal_cfg, preprocess_func=None
):
    conv_dict = {}
    for key in keys_list:
        # 1
        conv_dict[f"{key}_conv"] = copy.deepcopy([source["question"], source[f"{key}"]])

    if "image" in source:
        image = source["image"]
        image = multimodal_cfg["image_processor"](image)
        for key in keys_list:
            conv_dict[f"{key}_conv"] = expand_image_token(
                conv_dict[f"{key}_conv"], multimodal_cfg
            )

    data_dict = {}
    for key in keys_list:
        data_dict[f"{key}_data_dict"] = preprocess_func(
            [conv_dict[f"{key}_conv"]], tokenizer
        )

        # 2
        data_dict[f"{key}_data_dict"] = dict(
            input_ids=data_dict[f"{key}_data_dict"]["input_ids"][0],
            labels=data_dict[f"{key}_data_dict"]["labels"][0],
        )
        data_dict[f"{key}_data_dict"]["image"] = image

    if "ref_win_logp" in source:
        for key in keys_list:
            data_dict[f"{key}_data_dict"][f"ref_{key}_logp"] = source[f"ref_{key}_logp"]
            data_dict[f"{key}_data_dict"][f"ref_{key}_avg_logp"] = source[
                f"ref_{key}_avg_logp"
            ]
            data_dict[f"{key}_data_dict"][f"ref_{key}_per_token_logp"] = source[
                f"ref_{key}_per_token_logp"
            ]

    return_list = []
    for key in keys_list:
        return_list.append(data_dict[f"{key}_data_dict"])
    return return_list


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        image_token_len,
        img_processor,
        use_im_start_end=True,
    ):

        self.data = data
        self.mm_cfg = {
            "image_processor": img_processor,
            "is_multimodal": True,
            "image_token_len": image_token_len,
            "use_im_start_end": use_im_start_end,
            "keep_image_tag": True,
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # 从数据集中获取指定索引的样本
        sample = self.data[index]

        # 提取并处理样本的元信息
        metainfo = {
            "origin_dataset": sample["origin_dataset"],  # 原始数据集
            "origin_split": (
                json.loads(
                    sample["origin_split"]
                )  # 如果origin_split是字符串，则解析为JSON
                if type(sample["origin_split"]) == str
                else sample["origin_split"]  # 否则直接使用
            ),
            "origin_idx": sample["idx"],  # 原始索引
            "image_id": sample["image_path"],  # 图像ID，这里用图像路径表示
        }

        # 构造问题
        question = {
            "from": "human",
            "value": f"<image>\n{sample['question']}",
        }  # 问题来源为人类，值为图像标记和问题文本
        win = {
            "from": "gpt",
            "value": sample["chosen"],
        }  # 答案来源为GPT，值为GPT选择的答案
        rej = {
            "from": "gpt",
            "value": sample["rejected"],
        }  # 答案来源为GPT，值为GPT拒绝的答案
        image_only = {
            "from": "gpt",
            "value": sample["image_only"],
        }
        question_only = {
            "from": "gpt",
            "value": sample["question_only"],
        }

        # 将图像字节转换为PIL图像
        image = bytes_to_PIL_image(sample["image"]["bytes"])

        # 格式化样本
        formated_sample = {
            "image": image,  # 图像
            "question": question,  # 问题
            "win": win,  # GPT选择的答案
            "rej": rej,  # GPT拒绝的答案
            "image_only": image_only,
            "question_only": question_only,
            "idx": sample["idx"],  # 索引
            "metainfo": metainfo,  # 元信息
        }

        # 预处理函数，部分应用，指定有图像
        preprocess_func = partial(preprocess_v1, has_image=True)

        # 编码多模态偏好样本
        return_tuple = encode_multimodal_preference_sample(
            formated_sample,  # 格式化后的样本
            self.tokenizer,  # 分词器
            self.mm_cfg,  # 多模态配置
            preprocess_func=preprocess_func,  # 预处理函数
        )

        # 返回拒绝和接受的数据字典
        return return_tuple

    def __len__(self):
        return len(self.data)


def preference_collator_fn(instances, pad_token_id):
    # 将instances中的win和rej实例分开
    instance_list = list(zip(*instances))
    batch_dict = {}
    for i in range(len(keys_list)):
        batch_dict[f"{keys_list[i]}_batch"] = SFT_collator_fn(
            instance_list[i], pad_token_id
        )

    # 3
    # concatenated_input_ids = concate_pad(
    #     win_batch["input_ids"], rej_batch["input_ids"], pad_token_id
    # )
    # concatenated_labels = concate_pad(win_batch["labels"], rej_batch["labels"], -100)
    # concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    batch = {}
    for key in keys_list:
        batch[f"{key}_input_ids"] = batch_dict[f"{key}_batch"]["input_ids"]
        batch[f"{key}_labels"] = batch_dict[f"{key}_batch"]["labels"]
        batch[f"{key}_attention_mask"] = batch_dict[f"{key}_batch"]["attention_mask"]
    batch["images"] = batch_dict["win_batch"]["images"]
    batch["pad_token_id"] = pad_token_id
    return batch


def get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=False):
    logp_dict = {}
    for key in keys_list:
        logp_dict[f"{key}_logp_list"] = []
        logp_dict[f"{key}_avg_logp_list"] = []
        logp_dict[f"{key}_per_token_logp_list"] = []

    with torch.inference_mode():
        idx = 0
        for batch in tqdm.tqdm(dataloader):
            for key in keys_list:
                input_ids = batch[f"{key}_input_ids"].cuda()
                labels = batch[f"{key}_labels"].cuda()
                attention_mask = batch[f"{key}_attention_mask"].cuda()
                images = batch["images"]

                (_, _, _, _, inputs_embeds, labels) = (
                    model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=labels,
                        images=images.to(dtype=torch.bfloat16, device="cuda"),
                    )
                )
                output = model.forward(
                    inputs_embeds=inputs_embeds,
                    labels=None,
                )
                per_token_logp, log_prob, average_log_prob = get_batch_logps(
                    output.logits, labels, return_all=True
                )

                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                log_prob = log_prob.tolist()
                average_log_prob = average_log_prob.tolist()

                logp_dict[f"{key}_logp_list"] += log_prob
                logp_dict[f"{key}_avg_logp_list"] += average_log_prob
                logp_dict[f"{key}_per_token_logp_list"] += per_token_logp

    return_list = []
    for key in keys_list:
        return_list.append(logp_dict[f"{key}_logp_list"])
        return_list.append(logp_dict[f"{key}_avg_logp_list"])
        return_list.append(logp_dict[f"{key}_per_token_logp_list"])
    return tuple(return_list)


def inference_logp(
    model,
    tokenizer,
    hf_data,
    cache_file,
    image_token_len,
    img_processor,
    use_im_start_end,
    is_llava15=False,
):
    # 将模型转换为bfloat16格式并移至cuda设备
    model = model.to(dtype=torch.bfloat16, device="cuda")

    # 初始化数据集
    dataset = PreferenceInferenceDataset(
        tokenizer=tokenizer,
        data=hf_data,
        image_token_len=image_token_len,
        img_processor=img_processor,
        use_im_start_end=use_im_start_end,
    )

    # 设置数据加载器的collate_fn，并初始化数据加载器
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=64,
        shuffle=False,
        sampler=InferenceSampler(len(dataset)),
    )

    # 获取模型的对数概率输出
    outputs = get_multimodal_sample_logps(
        model, dataloader, tokenizer, is_llava15=is_llava15
    )  # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        try:
            torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
            merged_outputs[i] = [
                _ for _ in itertools.chain.from_iterable(merged_outputs[i])
            ]
        except Exception as e:
            print(
                f"Error during all_gather_object on rank {torch.distributed.get_rank()}: {e}",
                file=sys.stderr,
            )
            raise

    # 4
    logps = list(zip(*merged_outputs))
    df = write_logp_to_preference_parquet(
        dataset.data, cache_file, logps, overwrite_logps=True
    )
    torch.distributed.barrier()

    del model
    return df


class BiasPODataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_dir: str,
        multimodal_cfg: dict,
        reference_model=None,
    ):
        super(BiasPODataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = RLAIFVBiasDataset(
            data_dir,
            reference_model,
            tokenizer,
            multimodal_cfg["image_token_len"],
            multimodal_cfg["image_processor"],
            multimodal_cfg["use_im_start_end"],
            is_llava15=True,
        )
        self.multimodal_cfg = multimodal_cfg
        self.multimodal_cfg["keep_image_tag"] = True

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        preprocess_func = partial(preprocess_v1, has_image=True)
        return_tuple = encode_multimodal_preference_sample(
            source,
            self.tokenizer,
            self.multimodal_cfg,
            preprocess_func=preprocess_func,
        )
        return return_tuple


@dataclass
class DataCollatorForBiasPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    train_mode: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)
        pad_token_id = self.tokenizer.pad_token_id

        instance_list = list(zip(*instances))
        instances_dict = {}
        for i in range(len(keys_list)):
            instances_dict[f"{keys_list[i]}_instances"] = instance_list[i]

        batch["beta"] = self.beta
        for key in keys_list:
            batch[f"ref_{key}_logp"] = torch.as_tensor(
                [x[f"ref_{key}_logp"] for x in instances_dict[f"{key}_instances"]]
            )
            batch[f"ref_{key}_avg_logp"] = torch.as_tensor(
                [x[f"ref_{key}_avg_logp"] for x in instances_dict[f"{key}_instances"]]
            )
            ref_per_token_logp = [
                torch.as_tensor(x[f"ref_{key}_per_token_logp"])
                for x in instances_dict[f"{key}_instances"]
            ]
            batch[f"ref_{key}_per_token_logp"] = torch.nn.utils.rnn.pad_sequence(
                ref_per_token_logp, batch_first=True, padding_value=0
            )
            input_ids = batch[f"{key}_input_ids"]

            assert (
                batch[f"ref_{key}_per_token_logp"].size(1) >= input_ids.size(1) - 1
            ), f"{batch[f'ref_{key}_per_token_logp'].size(1)} >= {input_ids.size(1) - 1}"

            batch[f"ref_{key}_per_token_logp"] = batch[f"ref_{key}_per_token_logp"][
                :, : input_ids.size(1) - 1
            ]

        # 6
        for key in keys_list:
            for ins in instances_dict[f"{key}_instances"]:
                assert len(ins["input_ids"]) == len(ins["labels"])

        raw_keys = [
            "images",
            "beta",
            "win",
            "rej",
            "pad_token_id",
        ]
        if self.train_mode != "-":
            for key in self.train_mode.split("-"):
                raw_keys.append(key)
        keys_to_delete = [
            key for key in batch.keys() if not any(k in key for k in raw_keys)
        ]
        for key in keys_to_delete:
            del batch[key]
        torch.cuda.empty_cache()
        batch["train_mode"] = self.train_mode
        return batch


#################################################################################################
# BiasTrainer


def collect_preference_metrics(
    beta,
    q_dict,
    metrics,
    train_mode,
    chosen_rewards_list,
    policy_win_logp,
    # policy_win_avg_logp,
    ref_win_logp,
    # ref_win_avg_logp,
    rejected_rewards_list,
    policy_rej_logp_list,
    # policy_rej_avg_logp_list,
    ref_rej_logp_list,
    # ref_rej_avg_logp_list,
    reward_accuracies_list,
    preprocess_func,
):
    metrics = {}
    metrics[f"rewards_chosen/chosen"] = preprocess_func(chosen_rewards_list[0])
    metrics[f"logps_chosen/chosen"] = preprocess_func(policy_win_logp)
    metrics[f"logps_chosen/ref_chosen"] = preprocess_func(ref_win_logp)
    # metrics[f"logps_chosen/ref_chosen_avg"] = preprocess_func(ref_win_avg_logp)
    # metrics[f"logps_chosen/chosen_avg"] = preprocess_func(policy_win_avg_logp)
    # metrics[f"rewards_chosen/chosen_avg"] = (
    #     metrics[f"logps_chosen/chosen_avg"] - metrics[f"logps_chosen/ref_chosen_avg"]
    # )

    # metrics[f"rewards_raw/rejected"] = preprocess_func(rejected_rewards_list[0])
    # metrics[f"logps_raw/rejected"] = preprocess_func(policy_rej_logp_list[0])
    # metrics[f"logps_raw/ref_rejected"] = preprocess_func(ref_rej_logp_list[0])
    # metrics[f"logps_raw/ref_chosen"] = preprocess_func(ref_win_logp)
    # metrics[f"rewards_raw/accuracies"] = preprocess_func(reward_accuracies_list[0])
    # metrics[f"rewards_raw/margins"] = (
    #     metrics[f"rewards_raw/chosen"] - metrics[f"rewards_raw/rejected"]
    # )

    if len(list(q_dict.keys())) != 0:
        for key in q_dict.keys():
            metrics[f"lq/{key}_q"] = preprocess_func(q_dict[key])

    for i in range(0, len(train_mode), 2):
        rej_mode = train_mode[i + 1]
        if "question_only_noisy" in train_mode[i + 1]:
            rej_mode = "question_only"
        else:
            rej_mode = train_mode[i + 1]

        metrics[f"logps_{rej_mode}/rejected"] = preprocess_func(
            policy_rej_logp_list[int(i / 2)]
        )
        # metrics[f"logps_{rej_mode}/rejected_avg"] = preprocess_func(
        #     policy_rej_avg_logp_list[int(i / 2)]
        # )
        metrics[f"logps_{rej_mode}/ref_rejected"] = preprocess_func(
            ref_rej_logp_list[int(i / 2)]
        )
        # metrics[f"logps_{rej_mode}/ref_rejected_avg"] = preprocess_func(
        #     ref_rej_avg_logp_list[int(i / 2)]
        # )
        metrics[f"rewards_{rej_mode}/accuracies"] = preprocess_func(
            reward_accuracies_list[int(i / 2)]
        )
        metrics[f"rewards_{rej_mode}/rejected"] = preprocess_func(
            rejected_rewards_list[int(i / 2)]
        )
        # metrics[f"rewards_{rej_mode}/rejected_avg"] = (
        #     metrics[f"logps_{rej_mode}/rejected_avg"]
        #     - metrics[f"logps_{rej_mode}/ref_rejected_avg"]
        # )

        if train_mode[i] == "win":
            metrics[f"rewards_{rej_mode}/margins"] = (
                metrics[f"rewards_chosen/chosen"]
                - metrics[f"rewards_{rej_mode}/rejected"]
            )
            # metrics[f"rewards_{rej_mode}/margins_avg"] = (
            #     metrics[f"rewards_chosen/chosen_avg"]
            #     - metrics[f"rewards_{rej_mode}/rejected_avg"]
            # )
        else:
            metrics[f"rewards_{rej_mode}/margins"] = (
                metrics[f"rewards_chosen/paraphrase"]
                - metrics[f"rewards_{rej_mode}/rejected"]
            )

    return metrics


def get_logps(data_dict, model, args, mode_list, is_minicpm=False, is_llava15=False):
    batch_size = data_dict["win_input_ids"].shape[0]
    pad_token_id = data_dict["pad_token_id"]
    concatenated_input_ids = []
    concatenated_labels = []
    concatenated_attention_mask = []
    policy_logp_list = []
    policy_avg_logp_list = []
    ref_logp_list = []
    ref_avg_logp_list = []
    image_list = []

    new_mode_list = []
    if "win" in mode_list:
        new_mode_list.append("win")
    for i in range(1, len(mode_list), 2):
        new_mode_list.append(mode_list[i])
    if "paraphrased_chosen" in mode_list:
        new_mode_list.append("paraphrased_chosen")
    mode_list = new_mode_list

    for mode in mode_list:
        concatenated_input_ids.extend(list(data_dict[f"{mode}_input_ids"]))
        concatenated_labels.extend(list(data_dict[f"{mode}_labels"]))
        concatenated_attention_mask.extend(list(data_dict[f"{mode}_attention_mask"]))

        ref_avg_logp = data_dict[f"ref_{mode}_avg_logp"]
        ref_logp = data_dict[f"ref_{mode}_logp"]
        if args.dpo_use_average:
            ref_logp = ref_avg_logp
        ref_logp_list.append(ref_logp)
        ref_avg_logp_list.append(ref_avg_logp.detach())

        images = data_dict["images"]
        if is_minicpm:
            data_dict[f"{mode}_context_ids"]
        image_list.append(images)
    concatenated_input_ids = torch.nn.utils.rnn.pad_sequence(
        concatenated_input_ids,
        batch_first=True,
        padding_value=pad_token_id,
    )
    concatenated_labels = torch.nn.utils.rnn.pad_sequence(
        concatenated_labels,
        batch_first=True,
        padding_value=-100,
    )
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)
    concatenated_images = torch.cat(image_list, dim=0)

    (_, _, _, _, concatenated_inputs_embeds, concatenated_labels) = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
    )

    # 7
    output = model(
        inputs_embeds=concatenated_inputs_embeds,
        labels=None,
    )

    log_prob, average_log_prob = get_batch_logps(
        output.logits, concatenated_labels, return_per_token_logp=False
    )
    if args.dpo_use_average:
        policy_logp = average_log_prob
    else:
        policy_logp = log_prob
    # 8
    for i in range(0, len(log_prob), batch_size):
        policy_logp_list.append(policy_logp[i : i + batch_size])
        policy_avg_logp_list.append(average_log_prob[i : i + batch_size].detach())

    with torch.no_grad():
        avg_logits_list = []
        for i in range(1, len(policy_avg_logp_list)):
            avg_logits_list.append(
                (
                    policy_avg_logp_list[0]
                    - ref_avg_logp_list[0]
                    - policy_avg_logp_list[i]
                    + ref_avg_logp_list[i]
                ).detach()
            )
        torch.cuda.empty_cache()

    return (
        policy_logp_list,
        policy_avg_logp_list,
        ref_logp_list,
        ref_avg_logp_list,
    )


class LLaVA15BiasPOTrainer(ZephyrTrainer):
    def dpo_lq_dyn_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_avg_logp: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_rejected_avg_logp: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_chosen_avg_logp: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_rejected_avg_logp: torch.FloatTensor,
        beta: float,
        alpha: float,
        loss_type: str,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        with torch.no_grad():
            if loss_type == "avglogp":
                pi_avg_logratios = policy_avg_logp - policy_rejected_avg_logp
                ref_avg_logratios = (
                    reference_chosen_avg_logp - reference_rejected_avg_logp
                )
                avg_logits = pi_avg_logratios - ref_avg_logratios
                global_mean = gather_and_do_mean(torch.tensor(avg_logits))
            else:
                global_mean = gather_and_do_mean(torch.tensor(logits))
            q = torch.clamp(
                2
                * (
                    1
                    - torch.sigmoid(torch.tensor(alpha * global_mean))
                    .detach()
                    .to("cuda")
                ),
                min=0.001,
                max=1.0,
            )

        losses = (1 - torch.sigmoid(beta * logits) ** q) / q
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return (losses, q), chosen_rewards, rejected_rewards

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs

        if data_dict["train_mode"] == "-":
            train_mode = ["win", "rej"]
        else:
            train_mode = ["win", "rej"] + data_dict["train_mode"].split("-")

        beta = data_dict.pop("beta")

        if self.args.use_dyn_q == "-":
            use_dyn_q = []
        else:
            use_dyn_q = self.args.use_dyn_q.split("-")
            alpha_list = self.args.alpha.split("-")
            loss_type_list = self.args.loss_type.split("-")

        policy_logp_list = []
        policy_avg_logp_list = []
        ref_logp_list = []
        ref_avg_logp_list = []
        (
            policy_logp_list,
            policy_avg_logp_list,
            ref_logp_list,
            ref_avg_logp_list,
        ) = get_logps(
            data_dict, model, self.args, mode_list=train_mode, is_llava15=True
        )

        # 8
        losses_list = []
        rejected_rewards_list = []
        chosen_rewards_list = []
        reward_accuracies_list = []
        policy_chosen_logp = policy_logp_list[0]
        policy_chosen_avg_logp = policy_avg_logp_list[0]
        ref_chosen_logp = ref_logp_list[0]
        ref_chosen_avg_logp = ref_avg_logp_list[0]
        chosen_rewards = None

        q_dict = {}
        for i in range(0, len(train_mode), 2):
            policy_rej_logp = policy_logp_list[int(i / 2 + 1)]
            policy_rej_avg_logp = policy_avg_logp_list[int(i / 2 + 1)]
            ref_rej_logp = ref_logp_list[int(i / 2 + 1)]
            ref_rej_avg_logp = ref_avg_logp_list[int(i / 2 + 1)]

            dyn_q_index = -1
            for j in range(len(use_dyn_q)):
                if use_dyn_q[j] == train_mode[i + 1]:
                    dyn_q_index = j
                    break

            if dyn_q_index != -1:
                alpha = float(alpha_list[dyn_q_index])
                loss_type = loss_type_list[dyn_q_index]
                (
                    (losses, q),
                    chosen_rewards,
                    rejected_rewards,
                ) = self.dpo_lq_dyn_loss(
                    policy_chosen_logp,
                    policy_chosen_avg_logp,
                    policy_rej_logp,
                    policy_rej_avg_logp,
                    ref_chosen_logp,
                    ref_chosen_avg_logp,
                    ref_rej_logp,
                    ref_rej_avg_logp,
                    beta,
                    alpha,
                    loss_type,
                )
                if "question_only_noisy" in train_mode[i + 1]:
                    q_dict["question_only"] = q
                else:
                    q_dict[train_mode[i + 1]] = q
            else:
                losses, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logp,
                    policy_rej_logp,
                    ref_chosen_logp,
                    ref_rej_logp,
                    beta,
                )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            losses_list.append(losses)
            rejected_rewards_list.append(rejected_rewards)
            reward_accuracies_list.append(reward_accuracies)
        chosen_rewards_list.append(chosen_rewards)

        # print(f"Loss weights are {self.args.loss_weight}")
        # if self.args.loss_weight == "dyn":
        #     loss_weight_list = []
        #     for i in range(len(rejected_rewards_list)):
        #         loss_weight_list.append(
        #             float(gather_and_do_mean(chosen_rewards - rejected_rewards_list[i]))
        #         )
        #     weight_sum = sum(loss_weight_list)
        #     loss_weight_list = [max(x / weight_sum, 0.01) for x in loss_weight_list]
        #     print(f"Loss weights are {loss_weight_list}")
        #     loss = float(loss_weight_list[0]) * losses_list[0]
        #     for i in range(1, len(losses_list)):
        #         loss += float(loss_weight_list[i]) * losses_list[i]
        if self.args.loss_weight != "-":
            loss_weight_list = self.args.loss_weight.split("-")
            loss = float(loss_weight_list[0]) * losses_list[0]
            for i in range(1, len(losses_list)):
                loss += float(loss_weight_list[i]) * losses_list[i]
            # print(f"Loss weights are {loss_weight_list}")
        else:
            loss = losses_list[0]
            for i in range(1, len(losses_list)):
                loss += losses_list[i]

        loss = loss.mean()

        metrics = {}
        metrics = collect_preference_metrics(
            beta,
            q_dict,
            metrics,
            train_mode,
            chosen_rewards_list,
            policy_chosen_logp,
            ref_chosen_logp,
            rejected_rewards_list,
            policy_logp_list[1:],
            ref_logp_list[1:],
            reward_accuracies_list,
            gather_and_do_mean,
        )
        self.log(metrics)

        return loss
