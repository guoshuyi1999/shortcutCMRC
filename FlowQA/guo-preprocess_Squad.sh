set -e


mkdir -p Squad2_data
cp ../data/converted/squad22_train.json Squad2_data
cp ../data/converted/squad22_dev.json Squad2_data
# cp ../data/converted/dev-attack.json Squad2_data
# cp ../data/converted/dev-shuffle.json Squad2_data

# output_dir="Squad2_data/no-prepend/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_Squad2.py
#      --threads 8
#      --train_file "Squad2_data/train.json"
#      --dev_file "Squad2_data/dev.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


# output_dir="Squad2_data/no-prepend/attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_Squad2.py
#      --threads 8
#      --train_file "../data/converted/train_v0.2.json"
#      --dev_file "../data/converted/val_v0.2-attack.json"
#      --output_dir "${output_dir}")
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
# ##########33
########加的：no-question  mask

# #no-question
# output_dir="Squad2_data/no-question/no-attack"
# if [[ ! -d "${output_dir}" ]]
# then
#     mkdir -p "${output_dir}"
# fi
# cmd=(python preprocess_Squad2.py
#      --threads 8
#      --input_ablation 5
#      --dev_file "Squad2_data/squad22_dev.json"
#      --output_dir "${output_dir}"
#      )
# echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
# ${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"

##mask
output_dir="Squad2_data/mask-log/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_Squad2.py
     --threads 8
     --input_ablation 1
     --input_ablation_dev 1
     --dev_file "Squad2_data/squad22_dev.json"
     --output_dir "${output_dir}"
     )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


output_dir="Squad2_data/mask-pro/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_Squad2.py
     --threads 8
     --input_ablation 2
     --input_ablation_dev 2
     --dev_file "Squad2_data/squad22_dev.json"
     --output_dir "${output_dir}"
     )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


output_dir="Squad2_data/mask-no-wh/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_Squad2.py
     --threads 8
     --input_ablation 4
     --input_ablation_dev 4
     --dev_file "Squad2_data/squad22_dev.json"
     --output_dir "${output_dir}"
     )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


output_dir="Squad2_data/mask-only-wh/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_Squad2.py
     --threads 8
     --input_ablation 3
     --input_ablation_dev 3
     --dev_file "Squad2_data/squad22_dev.json"
     --output_dir "${output_dir}"
     )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
