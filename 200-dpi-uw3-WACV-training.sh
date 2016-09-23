#!/bin/bash
set -x
set -a
report_every=100
load=
save_every=1000
ntrain=1000000
dewarp=center
display_every=1000
test_every=1000
nhidden=100
lrate=1e-4
save_name=200-dpi-WACV
report_time=1000
./clstmocrtrain 200-dpi-book-1020-train.txt 200-dpi-book-1020-test.txt
