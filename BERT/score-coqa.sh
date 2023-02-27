set -e
cd src/


# Table 1 dev-orignal
for seed in 526
do
    for model in {3-pre,bert,mask-log,mask-pro,mask-no-wh,mask-only-wh,no-text,no-conv,no-position,no-qt,only-at-1,only-frist,only-second,only-third}; do
        model_dir="../bert-models-coqa/${seed}/${model}"
        cmd=(python scorer-coqa.py
             --val_file "../../data/converted/coqa_dev.json"
             --model_output "${model_dir}/predict-2.json"
             --o "${model_dir}/score-2.json"
            )
        echo "Running command ${cmd[@]}"
        ${cmd[@]}
    done
done

# #dev-attack
# for seed in 526
# do
#     for model in {3-pre,bert,mask-log,mask-pro,mask-no-wh,mask-only-wh,no-text,no-conv,no-position,no-qt,only-at-1,only-frist,only-second,only-third}; do
#         model_dir="../bert-models-coqa/${seed}/${model}"

#         # Table 4: with attack
#         cmd=(python scorer-coqa.py
#              --val_file "../../data/converted/coqa_dev.json"
#              --model_output "${model_dir}/predict-attack-2.json"
#              --o "${model_dir}/score-attack-2.json"
#             )
#         echo "Running command ${cmd[@]}"
#         ${cmd[@]}
        
#         # # Table 5: with mask
#         # cmd=(python scorer-coqa.py
#         #      --val_file "../../data/coqa/val_v0.2.json"
#         #      --model_output "${model_dir}/predict-mask-2.json"
#         #      --o "${model_dir}/score-mask-2.json"
#         #     )
#         # echo "Running command ${cmd[@]}"
#         # ${cmd[@]}
#     done

#     # # Table 6: with position information
#     # model_dir="../models/${seed}/bert"
#     # cmd=(python scorer-coqa.py
#     #      --val_file "../../data/coqa/val_v0.2.json"
#     #      --model_output "${model_dir}/predict-rm-2.json"
#     #      --o "${model_dir}/score-rm-2.json"
#     #     )
#     # echo "Running command ${cmd[@]}"
#     # ${cmd[@]}
# done

