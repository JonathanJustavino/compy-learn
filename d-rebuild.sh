#!/usr/bin/env bash

docker-compose down
rm -rf .docker
docker-compose build
