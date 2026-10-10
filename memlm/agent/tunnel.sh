#!/bin/sh
# Keep the GPU box's localhost:7777 forwarded to the kb bridge on this Mac.
# wssh multiplexes sessions (ControlMaster, its own -o comes first so it can't be overridden), so a
# `-R` forward lives on the shared master connection and `ssh -N` returns at once. Instead of a
# reconnect loop, health-check the remote port and (re)add the forward only when it is gone.
W="$HOME/wssh/wssh"
while true; do
  if ! "$W" ssh bmo@windows 'timeout 10 bash -c "</dev/tcp/127.0.0.1/7777"' 2>/dev/null; then
    echo "$(date) forward missing, re-adding" >&2
    "$W" ssh -O cancel -R 7777:localhost:7777 bmo@windows 2>/dev/null
    "$W" ssh -f -N -R 7777:localhost:7777 bmo@windows
  fi
  sleep 60
done
