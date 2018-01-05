#!/bin/bash

declare -a veclr=("1e-2" "5e-3" "1e-3" "5e-4", "1e-4")
declare -a vecmt=("9e-1" "95e-2" "99e-2")
declare -a vecwd=("1e-3" "1e-4" "1e-5")
declare -a vecbs=("64" "32")

for lr in "${veclr[@]}"; do
for mt in "${vecmt[@]}"; do
for wd in "${vecwd[@]}"; do
for bs in "${vecbs[@]}"; do
    echo "python try_resnet50_pretrained.py --lr $lr --mt $mt --wd $wd --bs $bs"
    python try_resnet50_pretrained.py --lr $lr --mt $mt --wd $wd --bs $bs
done; done; done; done
