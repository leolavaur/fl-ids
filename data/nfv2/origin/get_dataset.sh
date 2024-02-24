# get_dataset.sh © phdcybersec 2022
#
# Distributed under terms of the MIT license.


DATASETS=(
    "nf_v2"
    # nsl_kdd
    # cic_ids_2017
    # unsw_nb15
)
DEST="."

# Dowwnload NetFlow v2 dataset
# Args:
#   $1: destination directory
#
# Reference
#   Sarhan, M., Layeghy, S. & Portmann, M. Towards a Standard Feature Set for
#   Network Intrusion Detection System Datasets. Mobile Netw Appl (2021).
#   https://doi.org/10.1007/s11036-021-01843-0 
# URL:
#   https://staff.itee.uq.edu.au/marius/NIDS_datasets/

dl_nf_v2() {
    TMPDIR=$(mktemp -d)
    echo "Downloading NetFlow v2 dataset..."
    curl -L -o "$TMPDIR/nf_v2.zip" "https://cloudstor.aarnet.edu.au/plus/s/Y4tLFbVjWthpVKd/download?path=%2F&files[]=NF-UNSW-NB15-v2.csv&files[]=NF-ToN-IoT-v2.csv&files[]=NF-CSE-CIC-IDS2018-v2.csv&files[]=NF-BoT-IoT-v2.csv&downloadStartSecret=hy9arg9o0fq"
    echo "Extracting archive..."
    unzip -d "$TMPDIR" "$TMPDIR/nf_v2.zip"
    echo "Compressing files..."
    gzip "$TMPDIR"/*.csv
    echo "Moving files to $DEST..."
    mv "$TMPDIR"/*.gz "$DEST"
    rm -r "$TMPDIR"
    echo "Done."
}



# usage
usage() {
    echo "Usage: $0 [-d dataset]* [-h]" 1>&2
    echo "  -d dataset: download <dataset> to '$DEST'" 1>&2
    echo "  -l: list all datasets" 1>&2
    echo "  -h: help" 1>&2
}

list_datasets() {
    echo "Available datasets:"
    for d in "${DATASETS[@]}"; do
        echo "  - $d"
    done
}

declare -a TARGET
# parse args
while getopts "d:lh" opt; do
    case "${opt}" in
        'd')
            TARGET+=("${OPTARG}")
            ;;
        'l')
            list_datasets
            exit 0
            ;;
        'h')
            usage
            exit 0
            ;;
        '?')
            usage
            exit 1
            ;;
    esac
done


# for each provided dataset
for key in "${!TARGET[@]}"; do
    # check if dataset is valid
    if [[ ! "${DATASETS[*]}" =~ "${TARGET[$key]}" ]]; then
        echo "Invalid dataset: ${TARGET[$key]}"
        # remove invalid dataset from TARGET
        unset TARGET[$key]
    fi

done

if [[ ${#TARGET[@]} -eq 0 ]]; then
    # if no valid dataset is provided, exit with error
    exit 1
else
    if [[ ! -d "$DEST" ]]; then
        mkdir "$DEST"
    fi
    for target in "${TARGET[@]}"; do
        # download dataset
        dl_"${target}"
    done
fi
