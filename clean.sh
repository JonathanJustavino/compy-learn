#!/usr/bin/env bash

# remove all temporary data
rm -rf .eggs
rm -rf .pytest_cache
rm -rf build
rm -rf ComPy.egg-info
rm -rf dist
rm -rf tests/bin
rm -f compy/representations/extractors/*.so
rm -f .coverage
rm -f coverage.xml
