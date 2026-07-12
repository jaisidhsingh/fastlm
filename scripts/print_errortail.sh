#!/bin/bash

detect_cluster() {
  case "$(hostname -f)" in
    *cluster.is*) echo "mpi" ;;
    *capella*)    echo "capella" ;;
    *alpha*)      echo "alpha" ;;
    *)            echo "unknown" ;;
  esac
}

find_errfile() {
  local job_id="$1"

  case "$(hostname -f)" in
    *cluster.is*)
      echo "/fast/jsingh/logs/fastlm/june/attn/err/job.${job_id}.err"
      ;;
    *capella*)
      echo "/horse/ws/jasi149i-fastlm/logs/june/err/job-${job_id}.err"
      ;;
    *alpha*)
      echo "/horse/ws/jasi149i-fastlm/logs/june/err/job-${job_id}.err"
      ;;
    *)
      return 1
      ;;
  esac
}

job_id="$1"

if [[ -z "$job_id" ]]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

CLUSTER_ID=$(detect_cluster)

echo $CLUSTER_ID

ERRFILE=$(find_errfile "$job_id")

if [[ ! -f "$ERRFILE" ]]; then
  echo "Log file not found: $ERRFILE"
  exit 1
fi

tail -n 20 "$ERRFILE"
