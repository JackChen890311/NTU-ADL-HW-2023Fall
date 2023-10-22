#!bin/bash
mkdir data
mkdir model_mc
mkdir model_qa
gdown -O data/context.json 1aMuCaYMwvx5fGq-grUSxVlgh4dJG5oEy
gdown -O data/test.json 1riD-etAi_HOvalD-YcMfMvt2OQgNdrhS
gdown -O data/valid.json 1MHJEp4Rg8_1Qn7uFWO-y3F887jPIDp1b
gdown -O model_mc/model_mc.zip 1vjpIdUcJSKe2T3eY6pBCqM5loVbMHy01
gdown -O model_qa/model_qa.zip 1gIO6ZklNXdoQLmX3j8xn01sWvt0q7Rgt

unzip model_mc/model_mc.zip -d model_mc
unzip model_qa/model_qa.zip -d model_qa