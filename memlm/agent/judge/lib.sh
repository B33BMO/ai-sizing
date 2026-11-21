# opsd: simulated services for the judgment-eval image. Units live in /etc/opsd/units/<name> (bash: DESCRIPTION,
# RUN_AS, EXEC, ARGS, LOG, RELOAD=yes|no, plus exported SIM_* for simd). State in /var/lib/opsd.
S=/var/lib/opsd; U=/etc/opsd/units
mkdir -p $S /run/opsd
now() { date '+%a %Y-%m-%d %H:%M:%S %Z'; }
act() { echo "$(date +%s) $*" >> $S/actions.log; }
unit_name() { local u=${1%.service}; [ "$u" = ssh ] && u=sshd; echo "$u"; }
pid_of() { local p; p=$(cat /run/opsd/$1.pid 2>/dev/null) && kill -0 "$p" 2>/dev/null && echo "$p"; }
start_unit() {  # start_unit name
  local u=$1
  [ -n "$(pid_of $u)" ] && return 0
  ( . $U/$u
    [ "$u" = sshd ] && export SIM_PORT=$(awk 'tolower($1)=="port"{p=$2} END{print p?p:22}' /etc/ssh/sshd_config)
    cd /
    setsid setpriv --reuid "${RUN_AS:-root}" --regid "${RUN_AS:-root}" --init-groups \
      bash -c 'exec -a "$0" /usr/local/lib/opsd/simd "$@"' "$EXEC" $ARGS </dev/null >/dev/null 2>&1 &
    echo $! > /run/opsd/$u.pid )
  date '+%a %Y-%m-%d %H:%M:%S %Z' > $S/since.$u
  sleep 0.3
}
stop_unit() {
  local u=$1 p; p=$(pid_of $u) || return 0
  kill "$p" 2>/dev/null; sleep 0.2; kill -9 "$p" 2>/dev/null
  rm -f /run/opsd/$u.pid
  ( . $U/$u; [ -n "$LOG" ] && [ -n "$STOP_MSG" ] && echo "$(date '+%Y-%m-%d %H:%M:%S') $STOP_MSG" >> "$LOG" )
}
