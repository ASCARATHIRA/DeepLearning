#! /bin/sh
curl -L https://api.github.com/repos/ASCARATHIRA/DeepLearning/tarball > DeepLearning.tar.gz
tar -xzvf DeepLearning.tar.gz
cp -r ASCARATHIRA-DeepLearning*/Assignment4_17CS91P01/logs* .
