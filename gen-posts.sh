#!/bin/sh

DIR=$(pwd)/_posts
FILENAME=$(TZ='Asia/Seoul' date '+%Y-%m-%d')-$1.md

touch $DIR/$FILENAME

echo "# Craete by gen-posts.sh" >>$DIR/$FILENAME
