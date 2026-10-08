#!/bin/sh
# keep the GPU box's localhost:7777 forwarded to the kb bridge on this Mac
while true; do
  "$HOME/wssh/wssh" ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -R 7777:localhost:7777 bmo@windows
  echo "$(date) tunnel dropped, reconnecting" >&2
  sleep 5
done
