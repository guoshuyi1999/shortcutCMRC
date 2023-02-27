set -e


function train_quac {
    set -e
    mode=${1}
    seed=${2:=1023}

    case ${mode} in
    # Table 1
	"flowqa")
	    flags="-e 10  --start_dialog_ctx 0 --explicit_dialog_ctx 2"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"3-pre")
	    flags="-e 10  --start_dialog_ctx 0 --explicit_dialog_ctx 3"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"0-pre")
	    flags="-e 10 --start_dialog_ctx 0 --explicit_dialog_ctx 0"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;

	
	"only-frist") #1-pre
	    flags="--start_dialog_ctx 0 --explicit_dialog_ctx 1"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"only-second")
	    flags="--start_dialog_ctx 1 --explicit_dialog_ctx 2"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"only-third")
	    flags="--start_dialog_ctx 2 --explicit_dialog_ctx 3"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"no-conv")
	    flags="--explicit_dialog_ctx 0 --no_dialog_flow --no_hierarchical_query"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
	"no-text")
	    flags="--mask_prev_ans --no_hierarchical_query --no_dialog_flow"
	    train_dir="Squad2_data/no-prepend/no-attack"
	    valid_dir="Squad2_data/no-prepend/no-attack"
	    ;;
    # # Table 4, 5
	# "no-position")
	#     flags="--explicit_dialog_ctx 0"
	#     train_dir="Squad2_data/no-prepend/no-attack"
	#     valid_dir="Squad2_data/no-prepend/no-attack"
	#     ;;


	
	*)
	    echo "No matched mode!"
	    exit 1
	    ;;
    esac
    model_dir="squad-models/${seed}/${mode}"

    if [[ ! -d "${model_dir}" ]]
    then
        mkdir -p "${model_dir}"
    else
	echo "Output directory ${model_dir} exist. Abort!"
	exit 2
    fi
    
    cmd=(python train_QuAC.py ${flags}
         --seed ${seed}
	     --model_dir "${model_dir}"
         --train_dir "${train_dir}"
         --dev_dir   "${valid_dir}"
	)
    echo "Running command ${cmd[@]}" | tee "${model_dir}/train-log.txt"
    ${cmd[@]} | tee -a "${model_dir}/train-log.txt"
}


# train_quac flowqa 1023
# train_quac flowqa 1024
# train_quac flowqa 1025



# train_quac 3-pre  1023
# train_quac 3-pre  1024
# train_quac 3-pre  1025


# train_quac 0-pre  1023
# train_quac 0-pre  1024
# train_quac 0-pre  1025


# train_quac only-frist 1023
# train_quac only-frist 1024
# train_quac only-frist 1025

# train_quac only-second 1023
# train_quac only-second 1024
# train_quac only-second 1025


# train_quac only-third 1023
# train_quac only-third 1024
# train_quac only-third 1025


# train_quac no-conv 1023
# train_quac no-conv 1024
# train_quac no-conv 1025

# train_quac no-text 1023
# train_quac no-text 1024
# train_quac no-text 1025

# train_quac no-position 1023
# train_quac no-position 1024
# train_quac no-position 1025

# train_quac flowqa 1023

# train_quac 0-pre  1023
# train_quac 3-pre  1023
# train_quac only-third 1023
# train_quac only-second 1023
# train_quac only-frist 1023

# train_quac no-conv 1023
train_quac no-text 1023
# train_quac no-position 1023
# train_quac only-third 1023