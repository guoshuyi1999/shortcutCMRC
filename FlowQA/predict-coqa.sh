set -e

function predict_coqa {
    set -e
    seed=$1
    model=$2
    mode=$3

    model_path=./coqa-models/${seed}/${model}/best_model.pt
    output_dir=./coqa-models/predict/${seed}/${model}/${mode}

    if [ "${model}" == "no-conv" ] || [ "${model}" == "no-text" ] #||：表示 或,意为二者或多着只要满足其中一个
    then
        prepend="no-prepend"
    elif [ "${model}" == "no-position-fix" ] #如果=no-position
    then
        prepend='no-position'
    else                                                        #否则，其他全部的情况
        prepend="prepend"
    fi

    case ${mode} in
        "attack")
            dev_dir=CoQA_data/${prepend}/attack
            ;;
        "no-attack")
            dev_dir=CoQA_data/${prepend}/no-attack
            ;;
        "masked")
            extra_flags="--mask_prev_ans"
            dev_dir=CoQA_data/no-prepend/no-attack
            ;;
        "remove-indicator")
            extra_flags="--remove_indicator"
            dev_dir=CoQA_data/${prepend}/no-attack
            ;;
        "shuffle")
            dev_dir=CoQA_data/${prepend}/shuffle
            ;;
    esac

    if [[ ! -d "${output_dir}" ]]
    then
        mkdir "${output_dir}"
    else
        echo "${output_dir} existed. Abort!"
        exit 2
    fi

    cmd=(python predict_CoQA.py
         --model "${model_path}"
         --dev_dir "${dev_dir}"
         -o "${output_dir}"
         ${extra_flags})
    echo "Runnig command \"${cmd[@]}\" " | tee "${output_dir}/predict-log.txt"
    ${cmd[@]} | tee -a "${output_dir}/predict-log.txt"
}
#function predict_coqa


# # TABLE 1     seed  model        mode
predict_coqa  1023  flowqa       no-attack
predict_coqa  1023  0-pre        no-attack
predict_coqa  1023  3-pre        no-attack

predict_coqa  1023  only-first       no-attack
predict_coqa  1023  only-second      no-attack
predict_coqa  1023  only-third       no-attack
predict_coqa  1023  only-at-1        no-attack


predict_coqa  1023  no-text      no-attack
predict_coqa  1023  no-conv      no-attack

predict_coqa  1023  no-position-fix  no-attack
predict_coqa  1023  no-qt            no-attack

# predict_coqa  1024  flowqa       no-attack


predict_coqa  1023  mask-pro         no-attack
predict_coqa  1023  mask-log         no-attack
predict_coqa  1023  mask-no-wh       no-attack
predict_coqa  1023  mask-only-wh     no-attack

# seed: 1023
# TABLE 4     seed  model          mode
predict_coqa  1023  flowqa         attack
predict_coqa  1023  no-position-fix  attack
# TABLE 5     seed  model          mode
predict_coqa  1023  flowqa           masked
predict_coqa  1023  no-position-fix  masked
# TABLE 6     seed  model            mode
predict_coqa  1023  flowqa           remove-indicator
predict_coqa  1023  no-position-fix  remove-indicator
# TABLE 7     seed  model            mode
predict_coqa  1023  no-position-fix  shuffle

# # seed: 1024
# # TABLE 4     seed  model            mode
# predict_coqa  1024  flowqa           attack
# predict_coqa  1024  no-position-fix  attack
# # TABLE 5     seed  model            mode
# predict_coqa  1024  flowqa           masked
# predict_coqa  1024  no-position-fix  masked
# # TABLE 6     seed  model            mode
# predict_coqa  1024  flowqa           remove-indicator
# predict_coqa  1024  no-position-fix  remove-indicator
# # TABLE 7     seed  model            mode
# predict_coqa  1024  no-position-fix  shuffle
                                     
# # seed: 1025                         
# # TABLE 4     seed  model            mode
# predict_coqa  1025  flowqa           attack
# predict_coqa  1025  no-position-fix  attack
# # TABLE 5     seed  model            mode
# predict_coqa  1025  flowqa           masked
# predict_coqa  1025  no-position-fix  masked
# # TABLE 6     seed  model            mode
# predict_coqa  1025  flowqa           remove-indicator
# predict_coqa  1025  no-position-fix  remove-indicator
# # TABLE 7     seed  model            mode
# predict_coqa  1025  no-position-fix  shuffle

