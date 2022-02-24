# 1 context 2 train 3 validation
mkdir -p multi_choice_dataset
python3.9 pre_multi.py \
    --context_file ${1} \
    --train_file ${2} \
    --validation_file ${3} \
    --do_train 

python3.9 train_multi.py \
    --train_file multi_choice_dataset/train.json \
    --validation_file multi_choice_dataset/validation.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --output_dir multi_choice

python3.9 predict_multi.py \
    --context_file ${1} \
    --test_file ${3} \
    --model_name_or_path ./multi_choice \
    --tokenizer_name hfl/chinese-roberta-wwm-ext

python3.9 pre_qa.py \
  --context_file ${1} \
  --train_file ${2} \
  --validation_file test_clean.json \
  --do_train \

python3.9 train_qa.py  

