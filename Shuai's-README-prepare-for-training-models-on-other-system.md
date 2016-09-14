##	How to install prerequist and training on other systems

As a practice to run it on Wen's system so we can train on different computers. I install pre-requist for Ubuntu 14.04

run following commands first:
  sudo apt-get install mercurial
  sudo apt-get install libprotobuf-dev protobuf-compiler
  sudo apt-get install libzmq3-dev libzmq3 libzmqpp-dev libzmqpp3 libpng12-dev
  cd /usr/local/include
  sudo hg clone http://bitbucket.org/eigen/eigen eigen3
(Here you may face problems that eigen3 already exists, if so: run rm -rf eigen3)
  sudo hg up tensorflow_fix && cd -
###	How to install libprotobuf9 on 14.04
http://security.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf9_2.6.1-1_amd64.deb
###	How to install g++ 4.9 on 14.04
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install g++-4.9
  cd /usr/bin
  sudo rm gcc g++ cpp
  sudo ln -s gcc-4.9 gcc
  sudo ln -s g++-4.9 g++
  sudo ln -s cpp-4.9 cpp





###
Command to redirect training and testing history to a file so we can check it later.
Replace the
bash run-uw3-500 | tee -a /home/i59034/Documents/OCR-Verisk/docs/Logs/uw3-500-test.txt
