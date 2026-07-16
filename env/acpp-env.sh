# Put the AdaptiveCpp toolchain on PATH for this shell, if it isn't already.
# Source it: `source env/acpp-env.sh`.
#
# Override ACPP_HOME for a non-default install location:
#   ACPP_HOME=/opt/adaptivecpp source env/acpp-env.sh
: "${ACPP_HOME:=$HOME/adaptivecpp}"

if ! command -v acpp >/dev/null 2>&1; then
  export PATH="$ACPP_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$ACPP_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
