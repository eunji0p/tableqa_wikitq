// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import 'tapex_ITR_column_wise_wtq.jsonnet';

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

local override = {
  "model_config": {
    "ModelClass": "ITRRagReduceMixModel",
    "QueryEncoderModelVersion": "$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/train/saved_model/step_7603/query_encoder",
    "min_columns": 1,
    "index_files": index_files,
    "label_map_path": label_map_path,
  },
  "data_loader": {
    "dataset_modules": {
      "module_dict":{
        "LoadWikiTQData": {
          "type": "LoadWikiTQData", "option": "default",
          "config": {
            "preprocess": ["split_table_by_mixed_combination"],
            "path": {
              "train": "TableQA_data/wtq/preprocessed_split_table_by_mixed_combination_train.arrow",
              "validation": "TableQA_data/wtq/preprocessed_split_table_by_mixed_combination_validation.arrow",
              "test": "TableQA_data/wtq/preprocessed_split_table_by_mixed_combination_test.arrow",
            }
          },
        },
      },
    },
  },
  "metrics": [
    {'name': 'compute_denotation_accuracy'},
  ],
};

std.mergePatch(base_env, override)
