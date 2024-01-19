#!/usr/bin/env bash

# This script runs the specified experiment on a remote experiment server.

usage() {
    echo """
Usage: remoterun.sh -e|--experiment <experiment> -t|--target <target>
Runs the specified experiment on the specified target.

Options:
    -e, --experiment <experiment>  The experiment to run.
    -t, --target <target>          The target to run the experiment on.
    -h, --help                     Show this help message.
"""
}

main() {
    # Parse the command line arguments.
    OPTS=$(getopt -n "$0" -o he:t: --long help,experiment:,target: -- "$@")

    eval set -- "$OPTS"

    while true; do

        case "$1" in
            -e|--experiment)
                EXPERIMENT="$2"
                shift 2
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -h|--help)
                usage 
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unsupported option $1."
                exit 1
                ;;
        esac
    done

    # Check that the experiment directory exists.
    if [ ! -d "$EXPERIMENT" ]; then
        echo "Experiment directory '$EXPERIMENT' does not exist."
        exit 1
    fi

    # Check that the experiment directory contains a valid experiment.
    if [ ! -f "$EXPERIMENT/config.yaml" ]; then
        echo "Experiment directory '$EXPERIMENT' does not contain a valid 'config.yaml'."
        exit 1
    fi

    # Check that the target is a valid ssh host to which we can connect.
    if ! ssh "$TARGET" exit 0; then
        echo "Target '$TARGET' is not a valid ssh host, or you cannot connect to it."
        echo "Hint: is your VPN up?"
        exit 1
    fi

    CURRENT=$(git branch --show-current)
    RELEASE="release"

    # switch to a new branch, create it if it doesn't exist
    {
        git stash
        git checkout -b "$NEWBRANCHNAME" 
        git add .
        git commit -am "auto commit"
        git push --set-upstream origin "$NEWBRANCHNAME"
    } > /dev/null 2>&1


    # run the experiment on the remote host
    ssh "$TARGET" bash << EOF
echo "Running experiment '$EXPERIMENT' on '$TARGET'..."
cd ~/Workspace/phdcybersec/fl-ids
git fetch
git checkout origin/$NEWBRANCHNAME
tmux new-session -d -t "auto-remoterun" "nix develop -c eiffel --config-dir $EXPERIMENT; exit" || echo "Tmux session already running.\nCheck it out with 'tmux attach -t auto-remoterun'."
EOF

    git checkout "$CURRENTBRANCH"
    git stash pop

}

main "$@"


