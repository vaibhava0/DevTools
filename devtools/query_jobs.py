#!/usr/bin/env python3

import argparse
import subprocess
from collections import defaultdict
from pprint import pprint


def getJobArraySize(job_id):
    job_array = job_id.split("[")[1].split("]")[0]

    # Handles the case where max jobs are specified.
    if "%" in job_array:
        return int(job_array.split("%")[1])

    result = 0
    for part in job_array.split(','):
        if '-' in part:
            a, b = part.split('-')
            result += len(range(int(a), int(b) + 1))
        else:
            result += 1
    return result

# Note: supports simple job arrays.
def getGpuRequest(job_id, num_nodes):
    num_gpus = int(num_nodes) * 8
    if "_[" in job_id:
        num_gpus *= getJobArraySize(job_id)
    return num_gpus


def querySlurm(job_tag):
    slurm_command = 'squeue -a -o "%u,%i,%D,%j" -S u,i'
    process = subprocess.run(slurm_command, shell=True, check=True, capture_output=True, text=True)
    jobs = process.stdout.split("\n")
    # Remove the last empty entry.
    jobs.pop()
    job_info = [job.split(",", 4) for job in jobs]
    request_per_user = defaultdict(int)
    total_request = 0
    job_tag = job_tag.lower()
    for info in job_info:
        if job_tag in info[3].lower():
            request = getGpuRequest(info[1], info[2])
            # print(f"Request {request} for record {info}")
            request_per_user[info[0]] += request
            total_request += request
    print(f"\nTotal GPU request is: {total_request}")

    print(f"\nGPU request per user is:")
    for user_id, request in sorted(request_per_user.items()):
        print(f"{user_id}\t{request}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_tag', type=str, default="#winvision",
                        help='tag used to filter job names')

    args = parser.parse_args()
    querySlurm(args.job_tag)


if __name__ == "__main__":
    main()
