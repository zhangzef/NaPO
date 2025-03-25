import json
from collections import defaultdict, OrderedDict
import pdb
from copy import deepcopy
from tqdm import tqdm
import argparse
import numpy as np
import os
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument(
    "-dp", "--data_path", dest="data_path", required=True, type=str, action="store"
)
parser.add_argument(
    "-mid",
    "--model_identifier",
    dest="model_identifier",
    required=True,
    type=str,
    action="store",
)
parser.add_argument(
    "-vt", "--vote_thres", dest="vote_thres", type=int, action="store", default=2
)
parser.add_argument(
    "-st", "--style", dest="style", type=str, action="store", default="all"
)
parser.add_argument("-rf", "--result_file", dest="result_file", type=str, default="")
parser.add_argument("-ckpt", "--ckpt", dest="ckpt", type=str, default="")
args = parser.parse_args()


PRED_KEY = f"{args.model_identifier}"
dir = os.listdir(args.data_path)
context_list = []
for file in dir:
    try:
        with open(os.path.join(args.data_path, file), "r") as f:
            data = json.load(f)
            context_list.extend(data)
    except:
        continue
# with open(args.data_path, 'r') as f:
#     context_list = json.loads(f.read())
# pprint(context_list[0])
concept2num_instance = defaultdict(int)
concept2num_image = defaultdict(int)
concept2scores = defaultdict(lambda: defaultdict(float))

if args.style == "photorealistic":
    considered_image_ids = ["0", "1", "2", "3"]
elif args.style == "illustration":
    considered_image_ids = ["4", "5", "6", "7"]
elif args.style == "cartoon":
    considered_image_ids = ["8", "9", "10", "11"]
else:
    considered_image_ids = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
    ]


for instance in context_list:
    good_images = []
    for image_id, vote in instance["aggregated_human_label_good_images"].items():
        if vote >= args.vote_thres and image_id in considered_image_ids:
            good_images.append(image_id)
            concept2num_image[instance["concept"]] += 1
            concept2num_image["total"] += 1
    if len(good_images) == 0:
        continue
    concept2num_instance[instance["concept"]] += 1
    concept2num_instance["total"] += 1

    if (
        instance[f"{PRED_KEY}_a1_image"]["answer"] == "True"
        and instance[f"{PRED_KEY}_a2_image"]["answer"] == "False"
    ):
        instance["a"] = 1
        a_pass = 1
    else:
        instance["a"] = 0
        a_pass = 0

    if (
        instance[f"{PRED_KEY}_b1"]["answer"] == "True"
        and instance[f"{PRED_KEY}_b2"]["answer"] == "False"
    ):
        instance["b"] = 1
        b_pass = 1
    else:
        instance["b"] = 0
        b_pass = 0

    if (
        instance[f"{PRED_KEY}_c1_image"]["answer"] == "True"
        and instance[f"{PRED_KEY}_c2_image"]["answer"] == "False"
    ):
        instance["c"] = 1
        if a_pass:
            c_pass = 1
        else:
            c_pass = 0
    else:
        instance["c"] = 0
        c_pass = 0

    if b_pass and c_pass:
        bc_pass = 1
    else:
        bc_pass = 0

    instance["d"] = []
    d_pass = []
    for image_id in good_images:
        if (
            instance[f"{PRED_KEY}_d1"][image_id]["answer"] == "True"
            and instance[f"{PRED_KEY}_d2"][image_id]["answer"] == "False"
        ):
            instance["d"].append(1)
            if bc_pass:
                d_pass.append(1)
            else:
                d_pass.append(0)
        else:
            instance["d"].append(0)
            d_pass.append(0)
    d_macro = sum(instance["d"]) / len(good_images)
    d_pass_macro = sum(d_pass) / len(good_images)

    concept2scores[instance["concept"]]["a"] += instance["a"]
    concept2scores[instance["concept"]]["b"] += instance["b"]
    concept2scores[instance["concept"]]["c"] += instance["c"]
    concept2scores[instance["concept"]]["d_macro"] += d_macro
    concept2scores[instance["concept"]]["d_micro"] += sum(instance["d"])

    concept2scores[instance["concept"]]["a_pass"] += a_pass
    concept2scores[instance["concept"]]["b_pass"] += b_pass
    concept2scores[instance["concept"]]["c_pass"] += c_pass
    concept2scores[instance["concept"]]["bc_pass"] += bc_pass
    concept2scores[instance["concept"]]["d_pass_macro"] += d_pass_macro
    concept2scores[instance["concept"]]["d_pass_micro"] += sum(d_pass)

    concept2scores["total"]["a"] += instance["a"]
    concept2scores["total"]["b"] += instance["b"]
    concept2scores["total"]["c"] += instance["c"]
    concept2scores["total"]["d_macro"] += d_macro
    concept2scores["total"]["d_micro"] += sum(instance["d"])

    concept2scores["total"]["a_pass"] += a_pass
    concept2scores["total"]["b_pass"] += b_pass
    concept2scores["total"]["c_pass"] += c_pass
    concept2scores["total"]["bc_pass"] += bc_pass
    concept2scores["total"]["d_pass_macro"] += d_pass_macro
    concept2scores["total"]["d_pass_micro"] += sum(d_pass)
    concept2scores["total"]["bc_pass_micro"] += bc_pass * len(good_images)

for concept, score_dict in concept2scores.items():
    concept2scores[concept]["a_pass_ratio"] = (
        concept2scores[concept]["a_pass"] / concept2num_instance[concept]
    )
    concept2scores[concept]["b_pass_ratio"] = (
        concept2scores[concept]["b_pass"] / concept2num_instance[concept]
    )
    concept2scores[concept]["c_pass_ratio"] = (
        concept2scores[concept]["c_pass"] / concept2scores[concept]["a_pass"]
        if concept2scores[concept]["a_pass"] != 0
        else 0
    )
    concept2scores[concept]["d_pass_ratio_macro"] = (
        concept2scores[concept]["d_pass_macro"] / concept2scores[concept]["bc_pass"]
        if concept2scores[concept]["bc_pass"] != 0
        else 0
    )
    concept2scores[concept]["d_pass_ratio_micro"] = (
        concept2scores[concept]["d_pass_micro"]
        / concept2scores[concept]["bc_pass_micro"]
        if concept2scores[concept]["bc_pass_micro"] != 0
        else 0
    )


concept2scores["total"]["a_pass_ratio"] = (
    concept2scores["total"]["a_pass"] / concept2num_instance["total"]
)
concept2scores["total"]["b_pass_ratio"] = (
    concept2scores["total"]["b_pass"] / concept2num_instance["total"]
)
concept2scores["total"]["c_pass_ratio"] = (
    concept2scores["total"]["c_pass"] / concept2scores["total"]["a_pass"]
    if concept2scores["total"]["a_pass"] != 0
    else 0
)
concept2scores["total"]["d_pass_ratio_macro"] = (
    concept2scores["total"]["d_pass_macro"] / concept2scores["total"]["bc_pass"]
    if concept2scores["total"]["bc_pass"] != 0
    else 0
)
concept2scores["total"]["d_pass_ratio_micro"] = (
    concept2scores["total"]["d_pass_micro"] / concept2scores["total"]["bc_pass_micro"]
    if concept2scores["total"]["bc_pass_micro"] != 0
    else 0
)

# print(f"num total contexts: {concept2num_instance['total']}")
# print(f"num total images: {concept2num_image['total']}")
print(f"\n{args.model_identifier} scores:\n")

result_str = ""

for k in [
    "a_pass_ratio",
    "b_pass_ratio",
    "c_pass_ratio",
    "d_pass_ratio_macro",
    "c",
    "d_macro",
]:
    v = concept2scores["total"][k] * 100
    if "ratio" in k:
        print(f"{v:.1f} ", end="& ")
        result_str += f"{v:.1f} & "
    elif "micro" in k:
        print(f"{v / concept2num_image['total']:.1f} ", end="& ")
        result_str += f"{v / concept2num_image['total']:.1f} & "
    else:
        print(f"{v / concept2num_instance['total']:.1f} ", end="& ")
        result_str += f"{v / concept2num_instance['total']:.1f} & "
print("\n")


if args.result_file != "":
    with open(args.result_file, "a") as f:
        f.write("\n----------------------------------\n\n")
        f.write(f"{args.ckpt}-vlind:\n")
        f.write(
            "a_pass_ratio\tb_pass_ratio\tc_pass_ratio\td_pass_ratio_macro\tc\td_macro\n"
        )
        f.write("%s\n" % result_str)
        f.write("\n----------------------------------\n\n")
