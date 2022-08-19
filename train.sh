#!/bin/bash
fairseq-train \
    data-bin/wmt16_roen \
    --teacher-path ./AR-roen.pt \
    --arch reranker \
    -s ro \
    -t en \
    --task translation \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 512 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ./results/reranked_wmt16_roen \
    --log-file ./results/reranked_wmt16_roen/log.txt \
    --tensorboard-logdir ./results/reranked_wmt16_roen/ \
    --patience 50 \
    --validate-interval 1 \
    --num-workers 10 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 10\
    --valid-per-epoch 10
