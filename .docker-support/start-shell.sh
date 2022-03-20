#!/usr/bin/env bash

source "$VENV/bin/activate"
"$COMPY_LEARN/.docker-support/restore-binaries.sh"
exec bash
