set -e


# output_dir="QuAC_data/no-prepend-n/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --train_file "QuAC_data/train.json"
#      --dev_file "QuAC_data/val_v0.2-nn.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"

# output_dir="QuAC_data/no-prepend/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --train_file "QuAC_data/train.json"
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
# output_dir="QuAC_data/no-prepend/attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --train_file "../data/converted/train_v0.2.json"
#      --dev_file "../data/converted/val_v0.2-attack.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
###########33
#########加的：no-question  mask

# #no-question
# output_dir="QuAC_data/no-question/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 5
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"

# ##mask
# output_dir="QuAC_data/mask-log/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 1
#      --input_ablation_dev 1
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


# output_dir="QuAC_data/mask-pro/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 2
#      --input_ablation_dev 2
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


# output_dir="QuAC_data/mask-no-wh/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 4
#      --input_ablation_dev 4
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


# output_dir="QuAC_data/mask-only-wh/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 3
#      --input_ablation_dev 3
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
# 8.18
# #drop_except_most_similar_sentences
# output_dir="QuAC_data/drop_exsis_dev/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 0
#      --input_ablation_dev 0
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
# ##8.21


# #option == "drop_question_overlaps"
# output_dir="QuAC_data/drop_question_overlaps/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 7
#      --input_ablation_dev 7
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
# #option == "mask_question_overlaps"



# output_dir="QuAC_data/mask_question_overlaps/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 8
#      --input_ablation_dev 8
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"

# #mask_except_question_overlaps
# output_dir="QuAC_data/mask_except_question_overlaps/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --input_ablation 9
#      --input_ablation_dev 9
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"

# output_dir="QuAC_data/no-prepend/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_QuAC.py
#      --threads 8
#      --train_file "QuAC_data/train.json"
#      --dev_file "QuAC_data/dev.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


output_dir="QuAC_data/no-prepend/qorder"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_QuAC.py
     --threads 8
     --train_file "../data/quac/train_v0.2.json"
     --dev_file "../data/quac/8.24/val_v0.2-qorder-shuffle.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
