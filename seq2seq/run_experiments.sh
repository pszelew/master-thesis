#!/bin/bash

configs[0]="config_adv_0.02_mult_1.00_latent_256_epochs_15.yaml"

# configs[0]="config_adv_0.02_mult_0.02_latent_256.yaml"
# configs[1]="config_adv_0.02_mult_0.10_latent_256.yaml"
# configs[2]="config_adv_0.02_mult_0.50_latent_256.yaml"
# configs[3]="config_adv_0.02_mult_1.00_latent_256.yaml"

# configs[4]="config_adv_0.10_mult_0.02_latent_256.yaml"
# configs[5]="config_adv_0.10_mult_0.10_latent_256.yaml"
# configs[6]="config_adv_0.10_mult_0.50_latent_256.yaml"
# configs[7]="config_adv_0.10_mult_1.00_latent_256.yaml"

# configs[8]="config_adv_0.50_mult_0.02_latent_256.yaml"
# configs[9]="config_adv_0.50_mult_0.10_latent_256.yaml"
# configs[10]="config_adv_0.50_mult_0.50_latent_256.yaml"
# configs[11]="config_adv_0.50_mult_1.00_latent_256.yaml"

# configs[12]="config_adv_1.00_mult_0.02_latent_256.yaml"
# configs[13]="config_adv_1.00_mult_0.10_latent_256.yaml"
# configs[14]="config_adv_1.00_mult_0.50_latent_256.yaml"
# configs[15]="config_adv_1.00_mult_1.00_latent_256.yaml"

for config in "${configs[@]}"
do
    echo "Starting training for the config file ${config}";
    python train_seq2seq.py -c $config;
done
