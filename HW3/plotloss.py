import json
import matplotlib.pyplot as plt


if __name__ == '__main__':
    statejson = 'output/step1000_8bits_lr/checkpoint-1000/trainer_state.json'
    with open(statejson, 'r') as f:
        data = json.load(f)

    data = data['log_history']
    
    per_step = list(range(100, 1001, 100))
    perplexity = [4.3272, 4.2119, 4.0973, 4.0841, 4.03670, 4.0419, 4.0322, 4.0332, 3.9943, 3.9941]
    
    steps = [d['step'] for d in data if 'loss' in d]
    losses = [d['loss'] for d in data if 'loss' in d]

    eval_steps = [d['step'] for d in data if 'eval_loss' in d]
    eval_losses = [d['eval_loss'] for d in data if 'eval_loss' in d]

    plt.plot(steps, losses, label='train loss')
    plt.plot(eval_steps, eval_losses, label='valid loss')
    plt.plot(per_step, perplexity, label='valid perplexity')
    plt.xlabel('Steps')
    plt.ylabel('Loss / Perplexity')
    plt.title('Loss & Perplexity Plot')
    plt.show()
    plt.legend()
    plt.savefig('plot.png')