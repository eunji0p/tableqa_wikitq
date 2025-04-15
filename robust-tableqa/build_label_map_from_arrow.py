import json
import os
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

def question_type_classifier(query, header, tokenizer, model, device):
    """
    모델 불러오기 및 입력 텍스트 분류 예측 수행
    """
    # 입력 텍스트 토큰화: 모델 입력 형식에 맞게 변환 ([CLS] text [SEP])
    encoding = tokenizer(query, header, return_tensors="pt", truncation=True, padding=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # 모델 추론 (with torch.no_grad()로 기울기 계산 방지)
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits

    # logits의 argmax를 취해 예측된 label (0 또는 1)을 반환
    predicted_label = logits.argmax(dim=-1).item()

    return predicted_label


def build_label_map(dataset, tokenizer, model, device):
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
        # 예제 구조: {'id': ..., 'question': ..., 'table': {'header': [...], ...}, ...}
        question_id = str(example.get('id'))
        question_text = example.get('question', "")
        table = example.get('table', {})
        header = table.get('header', "")
        if isinstance(header, list):
            # 헤더 리스트를 문자열로 변환 (" * "로 join)
            header_str = " * ".join(str(h) for h in header)
        else:
            header_str = str(header)
        
        # 분류 (더미)
        label = question_type_classifier(question_text, header_str, tokenizer, model, device)
        label_map[question_id] = label

    return label_map

def main(args):

    model_name = "EUNJI0P/bert-query-type-cls-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # 평가 모드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    splits_label_map = {}

    for split in ["train", "validation", "test"]:
        file_path = os.path.join(args.data_dir, f"preprocessed_split_table_by_mixed_combination_{split}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{split} 파일({file_path})을 찾을 수 없습니다.")
        print(f"Loading {split} dataset from {file_path}...")
        ds = load_from_disk(file_path)
        print(f"Loaded {split} dataset with {len(ds)} examples.")
        splits_label_map[split] = build_label_map(ds, tokenizer, model, device) 
    
    # 최종적으로 모든 split의 label_map을 JSON으로 저장
    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump(splits_label_map, f, indent=4)
    print(f"Label maps saved to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build label maps for each split (train, validation, test) from Arrow datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory path where all Arrow datasets are stored"
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
python build_label_map.py \
  --data_dir /home/eunji/workspace/kim-internship/Eunji/Data/TableQA_data/wtq \
  --out_file /home/eunji/workspace/kim-internship/Eunji/Data/wtq_label_maps.json
"""
