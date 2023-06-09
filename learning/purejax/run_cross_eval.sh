#!/bin/bash

OUT_FILE="./logdir/cross_eval_results.csv"
CKPT_ROOT="./logdir/all_with_new_rew_ver3"

SYM_ENVS=(
    identical_5_vs_5
    identical_10_vs_10
    identical_20_vs_20
    special_5_vs_5
    special_10_vs_10
    special_20_vs_20
)

ASYM_ENVS=(
    identical_20_vs_5
    identical_10_vs_3
    identical_5_vs_1
    special_20_vs_5
    special_10_vs_3
    special_5_vs_1
)

CKPT_RANGE=$(seq 2000000 2000000 20000000)

### [SYM-ENV] team1 policy vs team1 policy
for ENV in ${SYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "11"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done

### [SYM-ENV] team2 policy vs team2 policy
for ENV in ${SYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "22"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done

### [ASYM-ENV] team1 policy vs team2 policy
for ENV in ${ASYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "12"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done

### [ASYM-ENV] team2 policy vs team1 policy
for ENV in ${ASYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "21"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done

### [SYM-ENV] team1 policy vs team2 policy
for ENV in ${SYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "12"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done

### [SYM-ENV] team2 policy vs team1 policy
for ENV in ${SYM_ENVS[@]}
do
    for CKPT1 in $CKPT_RANGE
    do
        for CKPT2 in $CKPT_RANGE
        do
            args=(
                --env-name ${ENV}_det
                --ckpt1 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT1}"
                --ckpt2 "${CKPT_ROOT}/${ENV}/ckpt/${CKPT2}"
                --ckpt-mode "21"
                --n-episodes 1000
                --min-ep-len 30
            )
            echo ${args[@]}
            python cross_eval.py ${args[@]} >> $OUT_FILE
        done
    done
done
