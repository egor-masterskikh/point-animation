SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

source "$PROJECT_DIR"/fonts_load.sh

context "$SCRIPT_DIR"/report.mkiv
