#!/usr/bin/env bash

docker-compose up -d
docker-compose exec compy-learn "/usr/share/compy-learn/.docker-support/start-shell.sh"
