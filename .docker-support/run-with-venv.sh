#!/usr/bin/env bash

source "$VENV/bin/activate"
exec ${@:-bash}
