from typing import Optional
import os
import argparse
import random
import json

import numpy as np
import pandas as pd


class TextInfo:
    def __init__(self, text: str, point: int = 0):
        self.text = text
        self.starts = []
        self.ends = []
        self.ents = []
        self.rels = []
        self.tags: Optional[list[str]] = None
        for word in text.split():
            end = point + len(word)
            self.starts.append(point)
            self.ends.append(end)
            point = end + 1  # 所有 word 之間都以一個空白隔開

    def create_tags(self):
        self.tags = ["O"] * len(self.starts)


def process_annotation(annotation: dict) -> tuple:
    text = annotation["value"]["text"]
    leading_whitespace_length = len(text) - len(text.lstrip())
    start = annotation["value"]["start"] + leading_whitespace_length
    end = start + len(text.strip())
    return text, start, end, annotation["value"]["labels"][0], annotation["id"]


def extract_annotation_info(json_data: list[dict], field_name: str) -> list[TextInfo]:
    label_data = []
    for example in json_data:
        if example["total_annotations"] != 1:
            continue
        example_info = TextInfo(text=example["data"][field_name])
        for annotation in example["annotations"][0]["result"]:
            if "value" in annotation:  # 如果有 "value" 的 key，代表為 entity 的標註，反之為 relation 的標註
                example_info.ents.append(process_annotation(annotation))
            else:
                example_info.rels.append((annotation["from_id"], annotation["to_id"]))
        label_data.append(example_info)
    return label_data


def train_test_split(*arrays, test_size: float = 0.25):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # 確保所有數組的長度相同
    length = len(arrays[0])
    for array in arrays:
        if len(array) != length:
            raise ValueError("All arrays must have the same length")

    # 生成隨機排列的索引
    indices = list(range(length))
    random.shuffle(indices)

    # 計算分割點
    split_idx = int(length * (1 - test_size))

    # 分割每個數組，並返回結果
    result = []
    for array in arrays:
        result.extend([np.array(array)[indices[:split_idx]].tolist(), np.array(array)[indices[split_idx:]].tolist()])
    return result


def convert_to_feature_data(label_data) -> list[tuple[str, str, str]]:
    feature_data = []
    for label in label_data:
        indices = []
        labels = []
        for ent in sorted(label.ents, key=lambda x: x[1]):
            indices.append(ent[1])
            labels.append(ent[3])
        feature_data.append((label.text, str(indices), str(labels)))
    return feature_data


def collect_tags(data) -> set[str]:
    tag_set = set()
    for example in data:
        for tag in eval(example[2]):
            tag_set.add(tag)
    return tag_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="This is annotated data for a NER task, coming from Label Studio in JSON format.")
    parser.add_argument("--field_name", type=str, required=True, help="The field name of the example in input_file.")
    parser.add_argument("--output_dir", type=str, required=True, help="The dataset for a NER task in CONLL format.")
    parser.add_argument("--seed", default=1314, type=int, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"The folder '{args.output_dir}' already exists!")

    with open(args.input_file) as f:
        json_data = json.load(f)

    label_data = extract_annotation_info(json_data, args.field_name)
    feature_data = convert_to_feature_data(label_data)

    train_data, test_data = train_test_split(feature_data, test_size=0.2)
    train_data, validation_data = train_test_split(train_data, test_size=0.125)

    train_tag_set = collect_tags(train_data)
    validation_tag_set = collect_tags(validation_data)
    test_tag_set = collect_tags(test_data)

    if validation_tag_set > train_tag_set or test_tag_set > train_tag_set:
        raise ValueError(f"The labels in the training data do not include those found in the validation and test data.")

    data_dir = f"{args.output_dir}/data"
    os.makedirs(data_dir)

    df_train = pd.DataFrame(train_data, columns=["Text", "Indices", "Tags"])
    df_validation = pd.DataFrame(validation_data, columns=["Text", "Indices", "Tags"])
    df_test = pd.DataFrame(test_data, columns=["Text", "Indices", "Tags"])

    df_train.to_csv(f"{data_dir}/train.csv", encoding="utf-8", index=False)
    df_validation.to_csv(f"{data_dir}/validation.csv", encoding="utf-8", index=False)
    df_test.to_csv(f"{data_dir}/test.csv", encoding="utf-8", index=False)


if __name__ == '__main__':
    main()
