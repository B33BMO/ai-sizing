#!/bin/sh
# keep the GPU box's localhost:7777 forwarded to the kb bridge on this Mac
while true; do
  # own connection (no ControlMaster mux): with mux the forward lands on the shared master and this
  # client exits at once, so the loop spins (7k reconnects in 2 days)
  "$HOME/wssh/wssh" ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -R 7777:localhost:7777 bmo@windows
  echo "$(date) tunnel dropped, reconnecting" >&2
  sleep 15
done
