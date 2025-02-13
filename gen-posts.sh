#!/bin/sh

TITLE="$1"
DATE=$(date "+%Y-%m-%d %H:%M:%S")
DIR=$(pwd)/_posts
FILENAME=$(TZ='Asia/Seoul' date '+%Y-%m-%d')-$TITLE.md

touch $DIR/$FILENAME

cat <<EOF >$DIR/$FILENAME
---
title: $TITLE
data: $DATE
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
---

# $TITLE

Craete by gen-posts.sh

EOF
