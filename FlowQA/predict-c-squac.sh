set -e

function predict_squad {
    seed=$1
    model=$2
    mode=$3
    threshold=$4
    
    model_path=./squad-models/${seed}/${model}/best_model.pt
    output_dir=./squad-models/predict/${seed}/${model}/${mode}
    #output_dir=./squad-models/${seed}/${model}/${mode}
    case ${mode} in
        "attack")
            dev_dir=Squad2_data/no-prepend/attack
            ;;
        "no-attack")
            dev_dir=Squad2_data/no-prepend/no-attack
            ;;
        "masked")
            extra_flags="--mask_prev_ans --explicit_dialog_ctx 2"
            dev_dir=Squad2_data/no-prepend/no-attack
            ;;
        "remove-indicator")
            extra_flags="--remove_indicator"
            dev_dir=Squad2_data/no-prepend/no-attack
            ;;
    esac

    if [[ ! -d "${output_dir}" ]]
    then
        mkdir "${output_dir}"
    else
        echo "${output_dir} existed. Abort!"
        exit 2
    fi

    cmd=(python predict_QuAC.py
         --model "${model_path}"
         --no_ans "${threshold}"
         --dev_dir "${dev_dir}"
         -o "${output_dir}/"
         ${extra_flags})
    echo "Runnig command \"${cmd[@]}\" " | tee "${output_dir}/predict-log.txt"
    ${cmd[@]} | tee -a "${output_dir}/predict-log.txt"
}

# #第四个参数不知道什么意思threshold，默认和flowqa一样吧
# TABLE 1     seed  model        mode             threshold
predict_squad  1023  flowqa       no-attack        -0.957
predict_squad  1023  no-text      no-attack        -1.357
predict_squad  1023  no-conv      no-attack        -0.896
predict_squad  1024  flowqa       no-attack           -0.360
predict_squad  1025  flowqa       no-attack           -0.960
predict_squad  1023  no-position  no-attack        -1.357
predict_squad  1024  no-position  no-attack        -0.435
predict_squad  1025  no-position  no-attack        -0.736


predict_squad  1023  0-pre        no-attack      -0.957
predict_squad  1023  3-pre        no-attack      -0.957

predict_squad  1023  only-first       no-attack  -0.957
predict_squad  1023  only-second      no-attack  -0.957
predict_squad  1023  only-third       no-attack  -0.957
predict_squad  1023  only-at-1        no-attack  -0.957


predict_squad  1023  no-position      no-attack  -0.957
predict_squad  1023  no-qt            no-attack  -0.957


predict_squad  1023  mask-pro         no-attack  -0.957
predict_squad  1023  mask-log         no-attack  -0.957
predict_squad  1023  mask-no-wh       no-attack  -0.957
predict_squad  1023  mask-only-wh     no-attack  -0.957



# seed: 1023
# TABLE 4     seed  model        mode             threshold
predict_squad  1023  flowqa       attack           -0.957
predict_squad  1023  no-position  attack           -1.357
# TABLE 5     seed  model        mode             threshold
predict_squad  1023  flowqa       mask             -0.957
predict_squad  1023  no-position  mask             -1.357
# TABLE 6     seed  model        mode             threshold
predict_squad  1023  flowqa       remove-indicator -0.957
predict_squad  1023  no-position  remove-indicator -1.357

# seed: 1024
# TABLE 4     seed  model        mode             threshold
predict_squad  1024  flowqa       attack           -0.360
predict_squad  1024  no-position  attack           -0.435
# TABLE 5     seed  model        mode             threshold
predict_squad  1024  flowqa       masked           -0.360
predict_squad  1024  no-position  masked           -0.435
# TABLE 6     seed  model        mode             threshold
predict_squad  1024  flowqa       remove-indicator -0.360
predict_squad  1024  no-position  remove-indicator -0.435

# seed: 1025
# TABLE 4     seed  model        mode             threshold
predict_squad  1025  flowqa       attack           -0.960
predict_squad  1025  no-position  attack           -0.736
# TABLE 5     seed  model        mode             threshold
predict_squad  1025  flowqa       mask             -0.960
predict_squad  1025  no-position  mask             -0.736
# TABLE 6     seed  model        mode             threshold
predict_squad  1025  flowqa       remove-indicator -0.960
predict_squad  1025  no-position  remove-indicator -0.736

