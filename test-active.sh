#!/bin/bash
set -x
set -a
report_every=100
save_every=500
ntrain=500000
dewarp=center
display_every=500
test_every=500
nhidden=100
target_height=32
lrate=1e-4
save_name=active-training
report_time=1000
./ocrActivetrain Thurs-dpi-book-1020-train.txt 5000 100 1e-3 0.02
