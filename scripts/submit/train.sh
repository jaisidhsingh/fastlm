#!/bin/bash

# tell us which cluster we're on
detect_cluster() {
  case "$(hostname -f)" in
    *cluster.is*) echo "mpi" ;;
    *capella*) echo "capella" ;;
    *alpha*) echo "alpha" ;;
    *)   echo "unknown" ;;
  esac
}

# map the path to our codebase each to available cluster
find_project() {
  case "$(hostname -f)" in
    *cluster.is*) echo "/home/jsingh/projects/fastlm" ;;
    *capella*) echo "/projects/p_neurasearch/fastlm" ;;
    *alpha*) echo "/projects/p_neurasearch/fastlm" ;;
    *)   echo "unknown" ;;
  esac
}

CLUSTER_ID=$(detect_cluster)
PROJECT=$(find_project)

echo $CLUSTER_ID $PROJECT

cd $PROJECT

arch_id=("gdn+attn_1-1" "gdn+attn_1-3")
n=("20M" "50M")
gbs=(64)
lr="all_parallel"
mode="main"
bid=100
submit="yes"
routine="train"

for aid in "${arch_id[@]}"; do
  for psid in "${n[@]}"; do
    for glbs in "${gbs[@]}"; do
      python -m manager \
        --cluster_id $CLUSTER_ID \
        --arch_id $aid \
        --n "$psid" \
        --gbs "$glbs" \
        --lr $lr \
        --mode $mode \
        --bid $bid \
        --submit $submit \
        --routine $routine;
    done
  done
done
