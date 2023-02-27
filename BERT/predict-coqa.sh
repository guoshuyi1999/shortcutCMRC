set -e
cd src/


# Table 1 #dev-orignal
#for seed in {526,525,526};
for seed in 526
do
    for model in {3-pre,bert,mask-log,mask-pro,mask-no-wh,mask-only-wh,no-text,no-conv,no-position,no-qt,only-at-1,only-frist,only-second,only-third}; do
        model_dir="../bert-models-coqa/${seed}/${model}"
        cmd=(python test.py
             "${model_dir}/config.json"
             "${model_dir}/model.pkl.2"
             "../data/coqa/coqa-bert/valid.pkl"
             "${model_dir}/predict-2.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-log-2.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-log-2.txt"
    done
done

##dev-attack
#for seed in {526,525,526};
for seed in 526
do
    for model in {3-pre,bert,mask-log,mask-pro,mask-no-wh,mask-only-wh,no-text,no-conv,no-position,no-qt,only-at-1,only-frist,only-second,only-third}; do
        model_dir="../bert-models-coqa/${seed}/${model}"

        # Table 4: with attack
        cmd=(python test.py
             "${model_dir}/config.json"
             "${model_dir}/model.pkl.2"
             "../data/coqa/bert-dev-attack/valid.pkl"
             "${model_dir}/predict-attack-2.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-attack-log-2.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-attack-log-2.txt"

        # Table 5: with mask
        # cmd=(python test.py
        #      "${model_dir}/config-mask.json"
        #      "${model_dir}/model.pkl.2"
        #      "../data/coqa-bert/valid.pkl"
        #      "${model_dir}/predict-mask-2.json"
        #     )
        # echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-mask-log-2.txt"
        # ${cmd[@]} | tee -a "${model_dir}/predict-mask-log-2.txt"
    done

    # # Table 6: without position information
    # model_dir="../bert-models-coqa/${seed}/bert"
    # cmd=(python test.py
    #      "${model_dir}/config-rm-ind.json"
    #      "${model_dir}/model.pkl.2"
    #      "../data/coqa-bert/valid.pkl"
    #      "${model_dir}/predict-rm-2.json"
    #     )
    # echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-rm-log-2.txt"
    # ${cmd[@]} | tee -a "${model_dir}/predict-rm-log-2.txt"
done

