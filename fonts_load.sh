PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [ "$OSFONTDIR" != "$PROJECT_DIR/fonts" ]; then
  OSFONTDIR="$PROJECT_DIR/fonts"
  export OSFONTDIR
#  mtxrun --generate
#  mtxrun --script font --reload
fi
