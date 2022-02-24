python3.9 predict_multi.py \
  --context_file ${1} \
  --test_file ${2} \
  --model_name_or_path ./multi_choice

python3.9 pre_qa.py \
  --context_file ${1} \
  --test_file ./test_clean.json \
  --do_test

python3.9 predict_qa.py  \
  --pred_file ${3}