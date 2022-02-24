import matplotlib.pyplot as plt
import json

models = ['1356', '2712', '4068', '5424', '6780', '8136', '9492', '10848', '12204', '13560', '14916', '16272', '17628', '18984', '20340', '21696']
rouge1, rouge2, rougel = [], [], []
steps = []
for model in models:
    data = json.load(open(f'results/result_{model}'))
    rouge1.append(data['rouge-1']['f'])
    rouge2.append(data['rouge-2']['f'])
    rougel.append(data['rouge-l']['f'])
    steps.append(int(model))
r1 = plt.plot(steps, rouge1, color='aquamarine', label="rouge-1")
r2 = plt.plot(steps, rouge2, color='gold', label="rouge-2")
rl = plt.plot(steps, rougel, color='red', label="rouge-l")
plt.legend(loc="lower right")
plt.xlabel('# of steps')
plt.ylabel('F1')

plt.savefig('result_f.png')
