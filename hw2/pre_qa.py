import json
import argparse

def processTest(f, context, data):
    for d in data:
        sub_d = {'id': d['id'], 'context': context[d['relevant']], 'question': d['question'], 'answers': [{"text":"", "start":-1}]}
        json.dump(sub_d, f, ensure_ascii=False)
        f.write('\n')
def processTrain(f, context, data):
    for d in data:
        sub_d = {'id': d['id'], 'context': context[d['relevant']], 'answers': d['answers'], 'question': d['question']}
        json.dump(sub_d, f, ensure_ascii=False)
        f.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_file", type=str, help="context file path"
    )
    parser.add_argument(
        "--train_file", type=str, help="train file path"
    )
    parser.add_argument(
        "--test_file", type=str, help="test file path"
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="validation file path"
    )
    parser.add_argument(
        "--do_train", action='store_true'
    )
    parser.add_argument(
        "--do_test", action='store_true'
    )
    args = parser.parse_args()
    return args

def main():
    global context
    args = parse_args()
    
    context = json.load(open(args.context_file, 'r'))

    if args.do_train:
        f_train = open('qa_dataset/train.json', 'w')
        train_data = json.load(open(args.train_file, 'r'))
        processTrain(f_train, context, train_data)
        if args.validation_file is not None:
            f_eval = open('qa_dataset/validation.json', 'w')
            eval_data = json.load(open(args.validation_file, 'r'))
            processTest(f_eval, context, eval_data)
    if args.do_test:
        f_test = open('qa_dataset/test.json', 'w')
        test_data = json.load(open(args.test_file, 'r'))
        processTest(f_test, context, test_data)

if __name__ == '__main__':
    main()