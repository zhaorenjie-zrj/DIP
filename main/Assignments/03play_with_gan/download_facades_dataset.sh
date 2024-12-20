#!/bin/bash
FILE=cityscapes
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE="/d/桌面/tuxiangchuli/assignment2/pix2/datasets/$FILE.tar.gz"
TARGET_DIR="/d/桌面/tuxiangchuli/assignment2/pix2/$FILE/"

mkdir -p "$TARGET_DIR"
echo "Downloading $URL to $TARGET_DIR"
wget -N "$URL" -O "$TAR_FILE"
mkdir -p "$TARGET_DIR"
tar -zxvf "$TAR_FILE" -C "/d/桌面/tuxiangchuli/assignment2/pix2/datasets/"
rm "$TAR_FILE"

find "${TARGET_DIR}train" -type f -name "*.jpg" | sort -V > "/d/桌面/tuxiangchuli/assignment2/pix2/train_list.txt"
find "${TARGET_DIR}val" -type f -name "*.jpg" | sort -V > "/d/桌面/tuxiangchuli/assignment2/pix2/val_list.txt"
