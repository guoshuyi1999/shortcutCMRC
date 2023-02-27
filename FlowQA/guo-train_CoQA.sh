set -e


function train_coqa {
    set -e
    mode=${1}
    seed=${2:=1023}

    case ${mode} in
    # Table 1
	"3-pre")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 3"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"0-pre")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 0"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"only-first")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 1"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"only-second")
	    flags="--start_dialog_ctx 1 --explicit_dialog_ctx 2"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"only-third")
	    flags="--start_dialog_ctx 2 --explicit_dialog_ctx 3"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
#############################################################
	"flowqa")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2"
	    train_dir="CoQA_data/prepend/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"no-conv")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 0 --no_dialog_flow --no_hierarchical_query"
	    train_dir="CoQA_data/no-prepend/no-attack"
	    valid_dir="CoQA_data/no-prepend/no-attack"
	    ;;
	"no-text")
	    flags="--mask_prev_ans --no_dialog_flow --no_hierarchical_query"
	    train_dir="CoQA_data/no-prepend/no-attack"
	    valid_dir="CoQA_data/no-prepend/no-attack"
	    ;;
    # Table 4, 5
	"no-position-fix")
	    flags="--explicit_dialog_ctx 0"
	    train_dir="CoQA_data/no-position/no-attack"
	    valid_dir="CoQA_data/no-position/no-attack"
	    ;;

	#################################
	"no-qt")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2 "
	    train_dir="CoQA_data/no-question/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"only-at-1")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 1"
	    train_dir="CoQA_data/no-question/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	############################

	"mask-pro")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2 "
	    train_dir="CoQA_data/mask-pro/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"mask-log")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2 "
	    train_dir="CoQA_data/mask-log/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;
	"mask-no-wh")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2 "
	    train_dir="CoQA_data/mask-no-wh/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;

	"mask-only-wh")
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 2 "
	    train_dir="CoQA_data/mask-only-wh/no-attack"
	    valid_dir="CoQA_data/prepend/no-attack"
	    ;;

	*)
	    echo "No matched mode!"
	    exit 1
	    ;;
    esac
    model_dir="coqa-models/${seed}/${mode}"

    if [[ ! -d "${model_dir}" ]]
    then
        mkdir -p "${model_dir}"
    else
	echo "Output directory ${model_dir} exist. Abort!"
	exit 2
    fi
    
    cmd=(python train_CoQA.py ${flags}
         --seed ${seed}
	     --model_dir "${model_dir}"
         --train_dir "${train_dir}"
         --dev_dir   "${valid_dir}"
	)
    echo "Running command ${cmd[@]}" | tee "${model_dir}/train-log.txt"
    ${cmd[@]} | tee -a "${model_dir}/train-log.txt"
}




# train_coqa flowqa  1023
# train_coqa flowqa  1024
# train_coqa flowqa  1025

# train_coqa no-conv 1023
# train_coqa no-conv 1024
# train_coqa no-conv 1025

# train_coqa no-text 1023
# train_coqa no-text 1024
# train_coqa no-text 1025

# train_coqa no-position-fix 1023
# train_coqa no-position-fix 1024
# train_coqa no-position-fix 1025



# train_coqa 3-pre  1023
# train_coqa 3-pre  1024
# train_coqa 3-pre  1025


# train_coqa 0-pre  1023
# train_coqa 0-pre  1024
# train_coqa 0-pre  1025

# train_coqa mask-log 1023
# train_coqa mask-log 1024
# train_coqa mask-log 1025

# train_coqa mask-pro 1023
# train_coqa mask-pro 1024
# train_coqa mask-pro 1025

# train_coqa mask-no-wh 1023
# train_coqa mask-no-wh 1024
# train_coqa mask-no-wh 1025

# train_coqa mask-only-wh 1023
# train_coqa mask-only-wh 1024
# train_coqa mask-only-wh 1025

# train_coqa no-qt 1023
# train_coqa no-qt 1024
# train_coqa no-qt 1025

# train_coqa only-at-1 1023
# train_coqa only-at-1 1024
# train_coqa only-at-1 1025


# train_coqa only-frist 1023
# train_coqa only-frist 1024
# train_coqa only-frist 1025

# train_coqa only-second 1023
# train_coqa only-second 1024
# train_coqa only-second 1025


# train_coqa only-third 1023
# train_coqa only-third 1024
# train_coqa only-third 1025


#############


# train_coqa flowqa  1023
# train_coqa no-conv 1023
# train_coqa no-text 1023
# train_coqa no-position-fix 1023
# train_coqa 3-pre  1023
# train_coqa 0-pre  1023
# train_coqa no-qt 1023
# train_coqa only-at-1 1023
# train_coqa mask-log 1023
# train_coqa mask-pro 1023
# train_coqa mask-no-wh 1023
# train_coqa mask-only-wh 1023
#######################
# train_coqa only-first 1023
# train_coqa only-second 1023
train_coqa only-third 1023



# train_coqa flowqa  1023

# train_coqa 3-pre  1023
# train_coqa 0-pre 1023
# train_coqa only-frist 1023
# train_coqa only-second 1023
# train_coqa only-third 1023

# train_coqa no-conv 1023
# train_coqa no-text 1023
# train_coqa no-position-fix 1023