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

arch_id=("attn" "gdn+attn_1-1" "gdn+attn_3-1" "gdn")
n="150M"
d="all_parallel"
gbs=32
lr=0.001
bid=100
submit="yes"

for aid in "${arch_id[@]}"; do
  python -m manager.eval \
    --arch_id $aid \
    --n $n \
    --d $d \
    --gbs $gbs \
    --lr $lr \
    --bid $bid \
    --submit $submit \
    --cluster_id $CLUSTER_ID;
done
