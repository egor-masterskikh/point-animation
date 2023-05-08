SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

source "$PROJECT_DIR"/fonts_load.sh

venv="$PROJECT_DIR"/venv

if [ ! -f "$SCRIPT_DIR/анимация_точки.pdf" ]; then
  source "$venv"/bin/activate
  "$venv"/bin/python "$SCRIPT_DIR"/main.py -s
fi

context "$SCRIPT_DIR"/report.mkiv
