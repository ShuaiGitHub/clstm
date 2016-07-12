#!/bin/bash
set -x
set -a

report_every=100
load=Pretrain-uw3-457000.clstm
save_every=1000
ntrain=1100000
#start = 1
dewarp=center
display_every=100
test_every=1000
display_every=1000
#testset=uw3-test.h5
nhidden=100
lrate=1e-4
save_name=Pretrain-uw3
report_time=1000
# gdb --ex run --args \
./clstmocrtrain uw3-norm-train uw3-norm-test
