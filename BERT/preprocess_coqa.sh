cd src/

# python make_dataset.py ../data/coqa/coqa-bert

# # printf "00"
# python make_dataset.py ../data/coqa/bert-dev-attack
# # printf "01"

#data/quac/bert-mask/wh/no-wh/train.pkl",pronouns

# python make_dataset.py ../data/coqa/bert-mask/logical

# python make_dataset.py ../data/coqa/bert-mask/pronouns

python make_dataset.py ../data/coqa/bert-mask/wh/only-wh

# python make_dataset.py ../data/coqa/bert-mask/wh/no-wh
# # # #dev




# set -e
# cd src/
# python make_dataset.py ../data/coqa/bert
# python make_dataset.py ../data/coqa/bert-dev-attack

# #dev
# #bert-dev-drop/logical,pronouns,wh drop dev

# python make_dataset.py ../data/coqa/bert-dev-drop/logical/all
# printf "1"
# python make_dataset.py ../data/coqa/bert-dev-drop/logical/context
# printf "2"
# python make_dataset.py ../data/coqa/bert-dev-drop/logical/conv
# printf "3"

# python make_dataset.py ../data/coqa/bert-dev-drop/pronouns/all
# printf "4"
# python make_dataset.py ../data/coqa/bert-dev-drop/pronouns/context
# printf "5"
# python make_dataset.py ../data/coqa/bert-dev-drop/pronouns/conv
# printf "6"

# python make_dataset.py ../data/coqa/bert-dev-drop/wh/all
# printf "7"
# python make_dataset.py ../data/coqa/bert-dev-drop/wh/context
# printf "8"
# python make_dataset.py ../data/coqa/bert-dev-drop/wh/conv
# printf "9"

# #bert-dev-mask/logical,pronouns,wh mask dev

# python make_dataset.py ../data/coqa/bert-dev-mask/logical/all
# printf "10"
# python make_dataset.py ../data/coqa/bert-dev-mask/logical/context
# printf "11"
# python make_dataset.py ../data/coqa/bert-dev-mask/logical/conv
# printf "12"


# python make_dataset.py ../data/coqa/bert-dev-mask/pronouns/all
# printf "13"
# python make_dataset.py ../data/coqa/bert-dev-mask/pronouns/context
# printf "14"
# python make_dataset.py ../data/coqa/bert-dev-mask/pronouns/conv
# printf "15"

# python make_dataset.py ../data/coqa/bert-dev-mask/wh/all
# printf "16"
# python make_dataset.py ../data/coqa/bert-dev-mask/wh/context
# printf "17"
# python make_dataset.py ../data/coqa/bert-dev-mask/wh/conv
# printf "18"


# #train
# #bert-train-drop/logical,pronouns,wh drop train

# python make_dataset.py ../data/coqa/bert-train-drop/logical/all
# printf "19"
# python make_dataset.py ../data/coqa/bert-train-drop/logical/context
# printf "20"
# python make_dataset.py ../data/coqa/bert-train-drop/logical/conv
# printf "21"


# python make_dataset.py ../data/coqa/bert-train-drop/pronouns/all
# printf "22"
# python make_dataset.py ../data/coqa/bert-train-drop/pronouns/context
# printf "23"
# python make_dataset.py ../data/coqa/bert-train-drop/pronouns/conv
# printf "24"

# python make_dataset.py ../data/coqa/bert-train-drop/wh/all
# printf "25"
# python make_dataset.py ../data/coqa/bert-train-drop/wh/context
# printf "26"
# python make_dataset.py ../data/coqa/bert-train-drop/wh/conv
# printf "27"

# #bert-train-mask/logical,pronouns,wh mask train

# python make_dataset.py ../data/coqa/bert-train-mask/logical/all
# printf "28"
# python make_dataset.py ../data/coqa/bert-train-mask/logical/context
# printf "29"
# python make_dataset.py ../data/coqa/bert-train-mask/logical/conv
# printf "30"


# python make_dataset.py ../data/coqa/bert-train-mask/pronouns/all
# printf "31"
# python make_dataset.py ../data/coqa/bert-train-mask/pronouns/context
# printf "32"
# python make_dataset.py ../data/coqa/bert-train-mask/pronouns/conv
# printf "33"

# python make_dataset.py ../data/coqa/bert-train-mask/wh/all
# printf "34"
# python make_dataset.py ../data/coqa/bert-train-mask/wh/context
# printf "35"
# python make_dataset.py ../data/coqa/bert-train-mask/wh/conv
# printf "36" 