#!/usr/bin/env bash

mkdir -p $COMPY_LEARN_BIN/tests/bin
cp -r $COMPY_LEARN_BIN/tests/bin $COMPY_LEARN/tests/bin 2>/dev/null || true

mkdir -p $COMPY_LEARN_BIN/compy/representations/extractors
cp $COMPY_LEARN_BIN/compy/representations/extractors/*.so $COMPY_LEARN/compy/representations/extractors 2>/dev/null || true
