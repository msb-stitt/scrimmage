#!/usr/bin/env bash
# Run a robot program in a Linux network namespace of its own, so its NetworkTables server has 5810 to itself
# (scrimmage/multi.py runs several robots on one machine). Run as root; the program runs as $SUDO_USER, or as RUN_AS.
#
#   tools/robot-netns.sh NAME INDEX -- COMMAND...
#
# NAME names the namespace, scr-NAME. INDEX, 1 to 250, gives it 10.231.INDEX.2, joined by a veth pair to the host's
# 10.231.INDEX.1: the program reaches the sim server at 10.231.INDEX.1 (HALSIMWS_HOST is set to it), and a dashboard
# on the host reads the robot's NetworkTables at 10.231.INDEX.2:5810. The namespace is removed when the program ends.
set -euo pipefail
[[ $# -ge 4 && $3 == "--" ]] || { echo "usage: $0 NAME INDEX -- COMMAND..." >&2; exit 2; }
name=$1 index=$2
shift 3
[[ $name =~ ^[A-Za-z0-9_-]{1,8}$ ]] || { echo "NAME: 1 to 8 letters, digits, _ or -" >&2; exit 2; }
[[ $index =~ ^[0-9]+$ ]] && ((index >= 1 && index <= 250)) || { echo "INDEX: 1 to 250" >&2; exit 2; }
user=${RUN_AS:-${SUDO_USER:-}}
[[ -n $user ]] || { echo "set RUN_AS to the user the robot program runs as" >&2; exit 2; }
ns=scr-$name host_if=scrh-$name ns_if=scrn-$name
host_ip=10.231.$index.1 ns_ip=10.231.$index.2

cleanup() { ip netns del "$ns" 2>/dev/null || true; ip link del "$host_if" 2>/dev/null || true; }
trap cleanup EXIT
cleanup
ip netns add "$ns"
ip link add "$host_if" type veth peer name "$ns_if"
ip link set "$ns_if" netns "$ns"
ip addr add "$host_ip/24" dev "$host_if"
ip link set "$host_if" up
ip -n "$ns" addr add "$ns_ip/24" dev "$ns_if"
ip -n "$ns" link set "$ns_if" up
ip -n "$ns" link set lo up
# in the background, so a TERM or INT reaches the robot program rather than waiting for it to end
ip netns exec "$ns" runuser -u "$user" -- env HALSIMWS_HOST="$host_ip" "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then   # a trapped signal ends the first wait early
  status=0
  wait "$child" || status=$?
fi
exit "$status"
