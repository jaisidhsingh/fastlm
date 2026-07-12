#!/bin/bash

detect_cluster() {
  case "$(hostname -f)" in
    *cluster.is*) echo "mpi" ;;
    *capella*)    echo "capella" ;;
    *alpha*)      echo "alpha" ;;
    *)            echo "unknown" ;;
  esac
}

find_logfile() {
  local job_id="$1"

  case "$(hostname -f)" in
    *cluster.is*)
      echo "/fast/jsingh/logs/fastlm/june/attn/out/job.${job_id}.out"
      ;;
    *capella*)
      echo "/horse/ws/jasi149i-fastlm/logs/june/out/job-${job_id}.out"
      ;;
    *alpha*)
      echo "/horse/ws/jasi149i-fastlm/logs/june/out/job-${job_id}.out"
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

LOGFILE=$(find_logfile "$job_id")

if [[ ! -f "$LOGFILE" ]]; then
  echo "Log file not found: $LOGFILE"
  exit 1
fi

tail -n 20 "$LOGFILE"
