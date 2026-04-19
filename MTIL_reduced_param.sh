CFG=mtil
DEVICE=1
EXP=MTIL_reduced_param

export PYTHONPATH="$PYTHONPATH:$PWD"


exp_name="exp/${EXP}"
SAVE_DIR=${exp_name}/domain-task/${CFG}

mkdir -p ${SAVE_DIR}
cp main.py ${SAVE_DIR}/
cp -r DPW ${SAVE_DIR}/
cp -r clip ${SAVE_DIR}/
cp clip/attention_reduced_param.py ${SAVE_DIR}/clip/attention.py
cp MTIL_reduced_param.sh ${SAVE_DIR}/


for SEED in 1 2 3
do
  CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=${SAVE_DIR}:$PYTHONPATH python ${SAVE_DIR}/main.py \
        --config-path configs/${CFG}.yaml \
        --seed ${SEED} \
        --log_path ${SAVE_DIR}/seed${SEED}
done
