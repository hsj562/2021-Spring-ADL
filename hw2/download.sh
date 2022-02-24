mkdir -p qa multi_choice qa_dataset
wget https://www.dropbox.com/s/sc2bsv7ug01k2mb/config.json?dl=1 -O multi_choice/config.json
wget https://www.dropbox.com/s/jmf3cdw0f2oalzy/pytorch_model.bin?dl=1 -O multi_choice/pytorch_model.bin
wget https://www.dropbox.com/s/4h2qczbra6vo978/special_tokens_map.json?dl=1 -O multi_choice/special_tokens_map.json
wget https://www.dropbox.com/s/0luo7tk60vqh2hr/tokenizer_config.json?dl=1 -O multi_choice/tokenizer_config.json
wget https://www.dropbox.com/s/a8wx75wx145l3vq/vocab.txt?dl=1 -O multi_choice/vocab.txt
wget https://www.dropbox.com/s/01fhv5dbarlfc6v/config.json?dl=1 -O qa/config.json
wget https://www.dropbox.com/s/stt55h2tlprz4jl/pytorch_model.bin?dl=1 -O qa/pytorch_model.bin
wget https://www.dropbox.com/s/vg5aza0k5w18qy8/special_tokens_map.json?dl=1 -O qa/special_tokens_map.json
wget https://www.dropbox.com/s/54yxphqatddmlur/tokenizer_config.json?dl=1 -O qa/tokenizer_config.json
wget https://www.dropbox.com/s/1ccee5xl5hxyz6t/training_args.bin?dl=1 -O qa/training_args.bin
wget https://www.dropbox.com/s/kphs53qjgn4l4q1/vocab.txt?dl=1 -O qa/vocab.txt