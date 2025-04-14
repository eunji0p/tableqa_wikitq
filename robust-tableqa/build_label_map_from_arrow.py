import json
import os
import argparse
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

def question_type_classifier(text):
    """
    모델 불러오기 
    """
    model_name = "EUNJI0P/bert-query-type-cls-model"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return random.choice([0, 1])


def build_label_map(dataset):
    """
    dataset: HuggingFace의 Arrow dataset (load_from_disk로 불러온)
    
    1. 각 예제에 대해 'question'과 'header' 정보를 결합하여 입력 text를 구성
    2. question_type_classifier 사용해 label을 예측합니다.
    3. 최종적으로 question_id를 key, 예측된 label (0 또는 1)을 value로 하는 label_map dict를 반환
    """
    # 결과 저장할 딕셔너리
    label_map = {}
    for example in dataset:
        # 예제에서 질문 및 헤더 정보 추출
        # 예제 구조: {'question_id': ..., 'question': ..., 'table': {'header': [...], ...}, ...}
        question_id = str(example.get('question_id', example.get('id')))
        question_text = example.get('question', "")
        table = example.get('table', {})
        header = table.get('header', "")
        if isinstance(header, list):
            # 헤더 리스트를 문자열로 변환 (" * "로 join)
            header_str = " * ".join(str(h) for h in header)
        else:
            header_str = str(header)
        
        # 질문과 헤더를 결합한 최종 입력 text 생성
        text_input = question_text + " " + header_str
        
        # 분류 (더미)
        label = question_type_classifier(text_input)
        label_map[question_id] = label
    return label_map

def main(args):
    splits_label_map = {}
    
    # 각 split의 arrow 파일을 개별적으로 불러오기
    for split, file_path in zip(["train", "validation", "test"], [args.train, args.validation, args.test]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{split} 파일({file_path})을 찾을 수 없습니다.")
        print(f"Loading {split} dataset from {file_path}...")
        ds = load_from_disk(file_path)
        print(f"Loaded {split} dataset with {len(ds)} examples.")
        # 해당 split의 label_map 생성
        splits_label_map[split] = build_label_map(ds)
    
    # 최종적으로 모든 split의 label_map을 JSON으로 저장
    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump(splits_label_map, f, indent=4)
    print(f"Label maps saved to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build label maps for each split (train, validation, test) from Arrow datasets."
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to the train arrow file (e.g., /path/to/train.arrow)"
    )
    parser.add_argument(
        "--validation",
        type=str,
        required=True,
        help="Path to the validation arrow file (e.g., /path/to/validation.arrow)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to the test arrow file (e.g., /path/to/test.arrow)"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="label_maps.json",
        help="Output JSON file path for the label maps."
    )
    
    args = parser.parse_args()
    main(args)

"""
python build_label_map_splits.py --train /path/to/train.arrow --validation /path/to/validation.arrow --test /path/to/test.arrow --out_file label_maps.json
"""
