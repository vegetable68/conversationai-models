#!/bin/bash

#
# A script to train the kaggle model locally
#

gcloud ml-engine local train \
     --module-name=trainer.model \
     --package-path=trainer \
     --job-dir=model -- \
     --train_data=train_small.csv \
     --predict_data=test.csv \
     --y_class=toxic \
     --train_steps=500 \
     --model=bag_of_words
