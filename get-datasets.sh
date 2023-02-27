

# # download original QuAC dataset
# mkdir -p data/quac/
# if [ ! -e "data/quac/train_v0.2.json" ]
# then
#     wget "https://s3.amazonaws.com/my89public/quac/train_v0.2.json" -O "data/quac/train_v0.2.json"
# fi
# if [ ! -e "data/quac/val_v0.2.json" ]
# then
#     wget "https://s3.amazonaws.com/my89public/quac/val_v0.2.json" -O "data/quac/val_v0.2.json"
# fi
# # apply the repeat attack on the QuAC
# python3 scripts/attack_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-attack.json"


# # download original CoQA dataset
# mkdir -p data/coqa/
# if [ ! -e "data/coqa/train.json" ]
# then
#     wget "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json" -O "data/coqa/train.json"
# fi
# if [ ! -e "data/coqa/dev.json" ]
# then
#     wget "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json" -O "data/coqa/dev.json"
# fi
# apply the repeat attack on the CoQA
# python3 scripts/attack_coqa.py "data/coqa/dev.json" "data/coqa/dev-attack.json"
# python3 scripts/shuffle_coqa.py "data/coqa/dev.json" "data/coqa/dev-shuffle.json"
# # python3 scripts/attack_coqa.py "data/coqa/dev.json" "data/coqa/dev-attack.json"
# python3 scripts/n_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-nn.json"
# python3 scripts/y_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-yy.json"
#
# python3 scripts/attack_qn_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-qnn.json"

# python3 scripts/attack_qn_quac.py "data/converted/coqa_dev.json" "data/converted/coqa_dev-qnn.json"


# #python3 scripts/attack_bn_coqa.py "data/converted/coqa_dev.json" "data/converted/coqa_dev-qnn.json"
# #python3 scripts/attack_coqa1.py "data/converted/coqa_dev.json" "data/converted/coqa_dev-qnn.json"
# python3 scripts/attack_bn_coqa.py "data/coqa/dev.json" "data/coqa/coqa_dev-qnn.json"
# python3 scripts/attack_sw_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-sww.json"
python3 scripts/attack_bsw_coqa.py "data/coqa/dev.json" "data/coqa/coqa_dev-sww.json"


#quac
#qnn sww

python3 scripts/attack_qn_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-qnn.json" #ok
python3 scripts/attack_sw_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-sww.json" #ok

#c-coqa
#qnn sww
python3 scripts/attack_qn_quac.py "data/converted/coqa_dev.json" "data/converted/coqa_dev-qnn.json" 
python3 scripts/attack_sw_quac.py "data/converted/coqa_dev.json" "data/converted/coqa_dev-qnn.json"



#coqa
#qnn sww
python3 scripts/attack_bn_coqa.py "data/coqa/dev.json" "data/coqa/coqa_dev-qnn.json"  #ok
python3 scripts/attack_bsw_coqa.py "data/coqa/dev.json" "data/coqa/coqa_dev-sww.json"    #ok


