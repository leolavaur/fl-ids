#!/usr/bin/env bash

# This script runs the specified experiment on a remote experiment server.

# TODO: Current state does not work.

usage() {
    echo """
Usage: remoterun.sh -e|--experiment <experiment> -t|--target <target>
Runs the specified experiment on the specified target.

Options:
    -e, --experiment <experiment>  The experiment to run.
    -t, --target <target>          The target to run the experiment on.
    -n, --notify-url <url>         The URL to notify when the experiment is done. (optional)
    -h, --help                     Show this help message.
"""
}

main() {
    # Parse the command line arguments.
    OPTS=$(getopt -o e:t:n:hd --long experiment:,target:,notify-url:,help,debug -n 'parse-options' -- "$@")

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
            -n|--notify-url)
                NOTIFYURL="$2"
                shift 2
                ;;
            -h|--help)
                usage 
                exit 0
                ;;
            -d|--debug)
                DEBUG=true
                shift
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

    # Config
    PUBLISHBRANCH="release"
    if [[ -n "$NOTIFYURL" ]]; then
        NOTIFYARGS="+callbacks=mattermost hydra.callbacks.mattermost.url=$NOTIFYURL"
    fi
    COMMAND="eiffel --config-dir $EXPERIMENT $NOTIFYARGS"

    if [[ -n "$DEBUG" ]]; then
        set -x

        # switch to a new branch, create it if it doesn't exist
        echo "Pushing current changes to the $PUBLISHBRANCH branch..."
        git commit -am ":rocket: Auto commit before running experiment '$EXPERIMENT' on '$TARGET'." || { echo "Could not commit changes. Aborting."; exit 1; }
        COMMIT=$(git rev-parse HEAD)
        git push --force origin "$COMMIT:$PUBLISHBRANCH" || { echo "Could not push changes. Aborting."; exit 1; }
        git reset --soft HEAD~1 || { echo "Could not reset changes. Aborting."; exit 1; }


        # run the experiment on the remote host
        echo "Running experiment '$EXPERIMENT' on '$TARGET'..."
        ssh "$TARGET" bash << EOF || { echo "Issue on remote host."; exit 1; } && echo "Experiment '$EXPERIMENT' running on '$TARGET', no apparent errors."
cd ~/Workspace/phdcybersec/fl-ids
git checkout "$PUBLISHBRANCH" || { echo "Branch '$PUBLISHBRANCH' does not exist. Aborting."; exit 1; }
git pull --rebase || { echo "Could not pull from remote. Aborting."; exit 1; }
tmux new-session -d -s "auto-remoterun" "nix develop -c $COMMAND || read && exit" || { echo "Tmux session already running."; echo "Check it out with 'tmux attach -t auto-remoterun'."; exit 1; } 
EOF
    else

        # switch to a new branch, create it if it doesn't exist
        echo "Pushing current changes to the $PUBLISHBRANCH branch..."
        git commit -am ":rocket: Auto commit before running experiment '$EXPERIMENT' on '$TARGET'." >/dev/null 2>&1 || { echo "Could not commit changes. Aborting."; exit 1; }
        COMMIT=$(git rev-parse HEAD)
        git push --force origin "$COMMIT:$PUBLISHBRANCH" >/dev/null 2>&1 || { echo "Could not push changes. Aborting."; exit 1; }
        git reset --soft HEAD~1 >/dev/null 2>&1 || { echo "Could not reset changes. Aborting."; exit 1; }


        # run the experiment on the remote host
        echo "Running experiment '$EXPERIMENT' on '$TARGET'..."
        ssh "$TARGET" bash << EOF || { echo "Issue on remote host."; exit 1; } && echo "Experiment '$EXPERIMENT' running on '$TARGET', no apparent errors."
cd ~/Workspace/phdcybersec/fl-ids
git checkout "$PUBLISHBRANCH" >/dev/null 2>&1 || { echo "Branch '$PUBLISHBRANCH' does not exist. Aborting."; exit 1; }
git pull --rebase >/dev/null 2>&1 || { echo "Could not pull from remote. Aborting."; exit 1; }
tmux new-session -d -s "auto-remoterun" "nix develop -c $COMMAND || read && exit" >/dev/null 2>&1 || { echo "Tmux session already running."; echo "Check it out with 'tmux attach -t auto-remoterun'."; exit 1; } 
EOF
    fi

}

main "$@"

