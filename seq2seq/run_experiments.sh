#!/bin/bash


configs[0]="config_adv_01.yaml"
configs[1]="config_adv_05.yaml"
configs[2]="config_adv_10.yaml"

for config in "${configs[@]}"
do
    echo "Starting training for the config file ${config}";
    python train_seq2seq.py -c $config;
done
