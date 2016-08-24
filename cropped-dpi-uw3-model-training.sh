#!/bin/bash
set -x
set -a
report_every=100
save_every=100
ntrain=1000000
dewarp=center
display_every=1000
test_every=1000
nhidden=100
lrate=1e-4
save_name=cropped-dpi
report_time=1000
./clstmocrtrain cropped-dpi-book-1020-train.txt cropped-dpi-book-1020-test.txt
