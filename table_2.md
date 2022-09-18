# Commando line to train Ditto model
## Table 2a
- Train with DistilBERT

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```

- Train with BERT  

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm bert   --fp16 --summarize --dk product   --da all```

- Train with RoBERTa

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm roberta   --fp16 --summarize --dk product   --da all```

- Train with XLNet

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm xlnet   --fp16 --summarize --dk product   --da all```

## Table 2b
- Sequence length: 32

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 32   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```
- Sequence length: 64

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```
- Sequence length: 128

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 128   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```
- Sequence length: 512

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 512  --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```

## Table 2c
- Ditto(plain)

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 ```

- Ditto(S)

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize```

- Ditto(S+DA)

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --da all```

- Ditto(S+KD)

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product```

- Ditto(S+DA+KD)

```CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task amazon_relaxdays_all_categories --batch_size 64 --max_len 64   --lr 3e-5  --finetuning --save_model  --logdir checkpoints/  --lm distilbert   --fp16 --summarize --dk product   --da all```