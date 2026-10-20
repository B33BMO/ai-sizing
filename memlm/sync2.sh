#!/bin/sh
# push local code to the GPU box (agent "windows", ~/memlm)
cd "$(dirname "$0")" && COPYFILE_DISABLE=1 tar cz --no-xattrs --exclude data --exclude runs --exclude .venv . | ${WSSH:-$(command -v wssh || echo "$HOME/wssh/wssh")} ssh bmo@windows 'mkdir -p ~/memlm2 && tar xz -C ~/memlm2'
