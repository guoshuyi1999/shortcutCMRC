set -e
cd src/


parent=${1:-"../bert-models-coqa"}

for seed in 526
do
    for mode in {mask-log,mask-pro,mask-no-wh,mask-only-wh,no-text,no-conv,no-position,no-qt,only-at-1,only-frist,only-second,only-third}
    do
	echo "Traing ${parent}/${seed}/${mode}..."
	python train.py "${parent}/${seed}/${mode}"
    done
done
#525 到 no-qt