import sys
import json
import jsonlines
import argparse
import os


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data


def save_json(json_path, data, indent=4):
    print("save json to {}".format(json_path))
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_all_jsonl_in_directory(directory):
    all_data = []
    dir_list = os.listdir(directory)
    dir_list.sort()
    for filename in dir_list:
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            print(file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                for item in file:
                    # reader = jsonlines.Reader(file)
                    # for item in reader:
                    all_data.append(json.loads(item))
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-template", type=str)
    parser.add_argument("--cap_folder", type=str)
    parser.add_argument("--save-file", type=str)

    args = parser.parse_args()

    print("======= merge review =========")
    print(args)

    path = args.response_template
    result_path = args.cap_folder
    save_path = args.save_file

    org_data = read_json(path)
    result_data = read_all_jsonl_in_directory(result_path)

    for i in range(len(org_data)):
        org_data[i]["model_answer"] = (
            result_data[i]["text"].replace("Assistant:", "").strip()
        )

    save_json(save_path, org_data)
