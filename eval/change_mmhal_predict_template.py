import sys
import json
import jsonlines
import argparse

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data

def save_json(json_path, data, indent=4):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_all_jsonl_in_directory(directory):
    all_data = []
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否是.jsonl文件
        if filename.endswith('.jsonl'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开.jsonl文件并读取数据
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = jsonlines.Reader(file)
                for item in reader:
                    all_data.append(item)
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response-template', type=str)
    parser.add_argument('--answers-file', type=str)
    parser.add_argument('--save-file', type=str)
    args = parser.parse_args()

    print("======= merge review =========")
    print(args)

    path = args.response_template
    result_path = args.answers_file
    save_path = args.save_file

    org_data = read_json(path)
    result_data = read_jsonl(result_path)

    for i in range(len(org_data)):
        org_data[i]["model_answer"] = result_data[i]["text"].replace("Assistant:", "").strip()

    save_json(save_path, org_data)