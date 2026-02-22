#!/bin/bash

INTERN_MODEL="InternVL3_5-2B"
ALGO="ICD"
#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_Base_55/${INTERN_MODEL}/${INTERN_MODEL}_MMStar_acc.csv
#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_Base_55/${INTERN_MODEL}/${INTERN_MODEL}_MME_score.csv
#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_Base_55/${INTERN_MODEL}/${INTERN_MODEL}_ScienceQA_VAL_acc.csv
#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_Base_55/${INTERN_MODEL}/${INTERN_MODEL}_RealWorldQA_acc.csv
#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_Base_55/${INTERN_MODEL}/${INTERN_MODEL}_BLINK_acc.csv

echo "✅ Extracting ${ALGO} results"
for SEED in 42 55 69
do
	echo "SEED ${SEED}"
	cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_${ALGO}_${SEED}/${INTERN_MODEL}/${INTERN_MODEL}_MMStar_acc.csv
	cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_${ALGO}_${SEED}/${INTERN_MODEL}/${INTERN_MODEL}_MME_score.csv
	cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_${ALGO}_${SEED}/${INTERN_MODEL}/${INTERN_MODEL}_ScienceQA_VAL_acc.csv
	cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_${ALGO}_${SEED}/${INTERN_MODEL}/${INTERN_MODEL}_RealWorldQA_acc.csv
	#cat outputs/${INTERN_MODEL}/${INTERN_MODEL}_${ALGO}_${SEED}/${INTERN_MODEL}/${INTERN_MODEL}_BLINK_acc.csv
done