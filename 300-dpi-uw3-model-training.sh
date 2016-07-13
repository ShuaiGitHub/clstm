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
save_name=300-dpi
report_time=1000
./clstmocrtrain Thurs-dpi-book-1020-train.txt Thurs-dpi-book-1020-test.txt
