#!/usr/bin/env bash

mkdir -p $COMPY_LEARN_BIN/tests/bin
cp -r $COMPY_LEARN/tests/bin $COMPY_LEARN_BIN/tests/bin

mkdir -p $COMPY_LEARN_BIN/compy/representations/extractors
cp $COMPY_LEARN/compy/representations/extractors/*.so $COMPY_LEARN_BIN/compy/representations/extractors/
