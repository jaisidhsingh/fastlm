#!/bin/bash

# tell us which cluster we're on
detect_cluster() {
  case "$(hostname -f)" in
    *cluster.is*) echo "mpi" ;;
    *capella*) echo "tud" ;;
    *alpha*) echo "tud" ;;
    *romeo*) echo "tud" ;;
    *barnard*) echo "tud" ;;
    *)   echo "unknown" ;;
  esac
}

# map the path to our codebase each to available cluster
find_project() {
  case "$(hostname -f)" in
    *cluster.is*) echo "/home/jsingh/projects/fastlm" ;;
    *capella*) echo "/projects/p_neurasearch/fastlm" ;;
    *alpha*) echo "/projects/p_neurasearch/fastlm" ;;
    *romeo*) echo "/projects/p_neurasearch/fastlm" ;;
    *barnard*) echo "/projects/p_neurasearch/fastlm" ;;
    *)   echo "unknown" ;;
  esac
}

CLUSTER_ID=$(detect_cluster)
PROJECT=$(find_project)

echo $CLUSTER_ID $PROJECT

cd $PROJECT

python -m services.sync --cluster_id $CLUSTER_ID
