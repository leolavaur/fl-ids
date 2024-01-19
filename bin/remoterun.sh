#!/usr/bin/env bash

# This script runs the specified experiment on a remote experiment server.

# Parse the command line arguments.
ARGS=$( getopt -o e:t: --long experiment:,target: -- "$@" )
eval set -- "$ARGS"

while true; do
    case "$1" in
        -e|--experiment)
            shift
            EXPERIMENT=$1
            shift
            ;;
        -t|--target)
            shift
            TARGET=$1
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
if ssh "$TARGET" exit; then
    echo "Target $TARGET is not a valid ssh host, or you cannot connect to it."
    echo "Hint: is your VPN up?"
    exit 1
fi

BRANCHNAME=auto-$(md5sum "$EXPERIMENT" | cut -d' ' -f1)

# switch to a new branch, create it if it doesn't exist
git checkout -b "$BRANCHNAME" || git checkout "$BRANCHNAME"
git commit -m "auto commit"
git push origin "$BRANCHNAME"

# run the experiment on the remote host
ssh "$TARGET" bash << 'EOF'
cd $HOME/Workspace/phdcybersec/fl-ids
git checkout origin/$BRANCHNAME
git pull 
tmux -d -t "auto-remoterun" <(eiffel --config-dir $EXPERIMENT; exit) || echo "Tmux session already running.\nCheck it out with 'tmux attach -t auto-remoterun'."
EOF





