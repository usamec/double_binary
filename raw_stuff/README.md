## Dependencies

```
torch==2.7.0
transformers==4.46.2
datasets==2.16.1
lm_eval==0.4.4
bf16_fused_adam==4.7.2
cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@24fbe4b5dab9a6c250a014573613c1890190536c
```

## Prepare RedPajama dataset

See prepare finetuning dataset in [AQLM repository](https://github.com/Vahe1994/AQLM?tab=readme-ov-file#preparing-fine-tuning-dataset).

## Compresion

See jupyter notebook in compress directory.

## Finetuning

See `run.sh` in finetuning directory.

## Evaluation

In `zeroshot_eval` directory:

Example: `python zeroshot_ours2.py --tasks $t --checkpoint latest_llama3inst-1b-15fin.pt`
