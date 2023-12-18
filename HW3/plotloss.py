import json
import matplotlib.pyplot as plt


if __name__ == '__main__':
    statejson = 'output/4bits/checkpoint-1000/trainer_state.json'
    with open(statejson, 'r') as f:
        data = json.load(f)

    data = data['log_history']
    
    per_step = list(range(100, 1001, 100))
    perplexity = [4.3857, 4.1677, 4.1285, 4.0843, 4.0755, 4.0106, 3.9645, 3.9821, 3.9325, 3.8961]
    
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
    plt.legend()
    plt.savefig('plot.png')