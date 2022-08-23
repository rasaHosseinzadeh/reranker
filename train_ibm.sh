#!/bin/bash
mkdir -p results/${1}/
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    /storage/sajad/data-bin/wmt14_deen \
    --teacher-path /storage/rasa/AR/WMTdeen.pt \
    --arch reranker \
    -s de \
    -t en \
    --task translation \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4  --lr-scheduler inverse_sqrt --warmup-updates 1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-tokens-valid 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ./results/${1}/ \
    --log-file ./results/${1}/log.txt \
    --tensorboard-logdir ./results/${1}/ \
    --patience 5 \
    --validate-interval 1 \
    --num-workers 40 \
    --keep-best-checkpoints 5 \
    --valid-per-epoch 10 \
    --fixed-validation-seed  7 \
    --teacher-ema-decay 1. \
    --teacher-beam 5 \
    --fp16
