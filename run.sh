#!/bin/sh

set -eu

exe=$1
shift

KMP_TEAMS_THREAD_LIMIT=1024 ./"$exe" "$@"
