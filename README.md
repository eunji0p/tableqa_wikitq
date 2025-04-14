# tableqa_wikitq

# Environment

```
pyenv create -n tableqa python=3.10.9
pyenv activate tableqa
```

## Main Experiments 

### ITR mix (intersecting columns and rows)

WikiSQL train
```
python src/main.py configs/wikisql/dpr_ITR_mix_wikisql.jsonnet --accelerator gpu --devices 8 --strategy ddp --num_sanity_val_steps 2 --experiment_name DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed --mode train --override --opts train.batch_size=1 train.scheduler=None train.epochs=20 train.lr=0.00001 train.additional.gradient_accumulation_steps=4 train.additional.warmup_steps=200 train.additional.early_stop_patience=8 train.additional.save_top_k=3 valid.batch_size=8 test.batch_size=8 valid.step_size=200 reset=1
```

WikiTQ test
```
python src/main.py configs/wtq/dpr_ITR_mix_wtq.jsonnet --accelerator gpu --devices 1 --strategy ddp --experiment_name DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed --mode test --test_evaluation_name wtq_original_sets --opts test.batch_size=32 test.load_epoch=11604
```

