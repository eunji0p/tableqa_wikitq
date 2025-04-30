# tableqa_wikitq

# Environment

```
pyenv create -n tableqa python=3.10.9
pyenv activate tableqa
```

## File


```    
    tableqa_wikitq/
                │
                ├── previous-robust-tableqa                  # 깃 클론한 원본 파일들
                ├── robust-tableqa                           # 실험에 맞게 수정 예정된 폴더
                ├── wtq_label_maps.json                      # 데이터셋 질문 유형 분류된 json 파일
```

## 📁 robust-tableqa 폴더


### config 파일 수정 (tapex_ITR_mix_wtq.jsonnet)

```    
    robust-tableqa/
                │ 
                ├── configs
                ├────── wtq
                ├──────── tapex_ITR_mix_wtq.jsonnet             

```

☑️ 아래부분 서버에 있는 파일 경로로 수정이 필요합니다!

* index_files는 아마 previous-robust-tableqa의 'Experiments' 폴더 안에 있을 가능성이 높습니다!

```

local index_files = {
  "index_paths": {
    "train": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/wtq_original_sets/step_7603/test.ITRWikiTQDataset.train",
    "validation": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/wtq_original_sets/step_7603/test.ITRWikiTQDataset.validation",
    "test": "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/wtq_original_sets/step_7603/test.ITRWikiTQDataset.test",
  },
};

// json 파일 주소
local label_map_path = {
  "label_map_file_path" : "/home/eunji/workspace/New_Eunji/wtq_label_maps.json",
};

```

* 최종 실행 코드 

☑️ 디바이스 수(8개) 확인이 필요합니다!

```
python src/main.py configs/wtq/tapex_ITR_mix_wtq.jsonnet --accelerator gpu --devices 8 --strategy ddp --num_sanity_val_steps 0 --experiment_name evaluate_ominitab_on_WTQ_15 --mode test --modules overflow_only original_sub_table_order --test_evaluation_name original_sets --opts test.batch_size=2 test.load_epoch=0 model_config.GeneratorModelVersion=neulab/omnitab-large-finetuned-wtq model_config.DecoderTokenizerModelVersion=neulab/omnitab-large-finetuned-wtq model_config.ModelClass=ITRRagReduceMixModel data_loader.additional.num_knowledge_passages=15
```

