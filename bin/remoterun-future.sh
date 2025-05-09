#!/usr/bin/env bash

# This script runs the specified experiment on a remote experiment server.

usage() {
    echo """
Usage: remoterun.sh 
Runs the specified experiment on the specified target.

Options:
    -e, --experiment <experiment>  The experiment to run.
    -t, --target <target>          The target to run the experiment on.
    -r, --rev <rev>                The git revision to get the eiffel environment from.
    -n, --notify-url <url>         The URL to notify when the experiment is done. (optional)
    -d, --debug                    Run in debug mode.
    -h, --help                     Show this help message.
"""
}

main() {
    # Parse the command line arguments.
    OPTS=$(getopt -o e:t:n:r:hd --long experiment:,target:,notify-url:,rev:,help,debug -n 'parse-options' -- "$@")

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
            -r|--rev)
                REV="$2"
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
    EIFFELUSER="phdcybersec"
    EIFFELREPO="eiffel"
    RET=$(curl -LsI -o /dev/null -w "%{http_code}" "https://github.com/$EIFFELUSER/$EIFFELREPO/commit/$REV") 
    if [[ $RET -ne 200 ]]; then
        echo "Could not find valid commit: 'https://github.com/$EIFFELUSER/$EIFFELREPO/commit/$REV'"
        exit 1
    fi
    FLAKE="github:$EIFFELUSER/$EIFFELREPO/$REV"

    PUBLISHBRANCH="release"
    if [[ -n "$NOTIFYURL" ]]; then
        NOTIFYARGS="+callbacks=mattermost hydra.callbacks.mattermost.url=$NOTIFYURL"
    fi
    COMMAND="eiffel --config-dir $EXPERIMENT $NOTIFYARGS"
    
    REPO=$(git rev-parse --show-toplevel) || { echo "Could not find the root of the git repository. Aborting."; exit 1; } 
    cd "$REPO" || { echo "Could not change to the root of the git repository. Aborting."; exit 1; }

    # Run the experiment
    if [[ -n "$DEBUG" ]]; then
        set -x

        # push the current changes to git
        echo "Pushing current changes to the $PUBLISHBRANCH branch..."
        git add .
        if [[ -n $(git status --porcelain) ]]; then
            git commit -am ":rocket: Auto commit before running experiment '$EXPERIMENT' on '$TARGET'." || { echo "Could not commit changes. Aborting."; exit 1; }
            COMMIT=$(git rev-parse HEAD)
            git push --force origin "$COMMIT:$PUBLISHBRANCH" || { echo "Could not push changes. Aborting."; exit 1; }
            git reset --soft HEAD~1 || { echo "Could not reset changes. Aborting."; exit 1; }
        else
            COMMIT=$(git rev-parse HEAD)
        fi

        RUNDIR="/tmp/eiffel/$COMMIT"
        ssh "$TARGET" mkdir -p "$RUNDIR/$EXPERIMENT" || { echo "Could not create the run directory on the remote host. Aborting."; exit 1; }
        # copy the experiment to the remote host
        echo "Copying experiment '$EXPERIMENT' to '$TARGET:$RUNDIR/$EXPERIMENT'..."
        rsync -azC --progress --exclude='.git' --exclude="*.pkl" --partial --inplace "$EXPERIMENT/" "$TARGET:$RUNDIR/$EXPERIMENT" || { echo "Could not copy the experiment to the remote host. Aborting."; exit 1; }

        # run the experiment on the remote host
        echo "Running experiment '$EXPERIMENT' on '$TARGET'..."
        echo "Running: $COMMAND"
        ssh "$TARGET" bash << EOF || { echo "Issue on remote host."; exit 1; } && echo "Experiment '$EXPERIMENT' running on '$TARGET', no apparent errors."
set -x
cd $RUNDIR
tmux new-session -d -s "auto-remoterun" "nix develop $FLAKE --command $COMMAND || read && exit" || { echo "Tmux session already running."; echo "Check it out with 'tmux attach -t auto-remoterun'."; exit 1; } 
EOF

    else
        # push the current changes to git
        echo "Pushing current changes to the $PUBLISHBRANCH branch..."
        git add .
        if [[ -n $(git status --porcelain) ]]; then
            git commit -am ":rocket: Auto commit before running experiment '$EXPERIMENT' on '$TARGET'."  >/dev/null 2>&1 || { echo "Could not commit changes. Aborting."; exit 1; }
            COMMIT=$(git rev-parse HEAD)
            git push --force origin "$COMMIT:$PUBLISHBRANCH" >/dev/null 2>&1 || { echo "Could not push changes. Aborting."; exit 1; }
            git reset --soft HEAD~1 >/dev/null 2>&1 || { echo "Could not reset changes. Aborting."; exit 1; }
        fi

        COMMIT=$(git rev-parse HEAD)
        RUNDIR="/tmp/eiffel/$COMMIT"
        ssh "$TARGET" mkdir -p "$RUNDIR/$EXPERIMENT" >/dev/null 2>&1 || { echo "Could not create the run directory on the remote host. Aborting."; exit 1; }
        # copy the experiment to the remote host
        echo "Copying experiment '$EXPERIMENT' to '$TARGET:$RUNDIR/$EXPERIMENT'..."
        rsync -azC --progress --exclude='.git' --exclude="*.pkl" --partial --inplace "$EXPERIMENT/" "$TARGET:$RUNDIR/$EXPERIMENT" >/dev/null 2>&1 || { echo "Could not copy the experiment to the remote host. Aborting."; exit 1; }

        # run the experiment on the remote host

        echo "Running experiment '$EXPERIMENT' on '$TARGET'..."
        ssh "$TARGET" bash << EOF >/dev/null 2>&1 || { echo "Issue on remote host."; exit 1; } && echo "Experiment '$EXPERIMENT' running on '$TARGET', no apparent errors."
cd $RUNDIR/$EXPERIMENT
tmux new-session -d -s "auto-remoterun" "nix develop $FLAKE --command $COMMAND || read && exit" || { echo "Tmux session already running."; echo "Check it out with 'tmux attach -t auto-remoterun'."; exit 1; }
EOF
    fi

}

main "$@"
cd - >/dev/null 2>&1 || exit 1

