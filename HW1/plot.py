import json
import matplotlib.pyplot as plt

def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    if 'legend' not in config or config['legend']:
        plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()

if __name__ == '__main__':
    # json_path = '/shared_home/r12922051/Code/NTU-ADL-HW-2023Fall/HW1/data/context.json'
    # with open(json_path,'r') as f:
    #     context = json.load(f)
    # print(len(context))
    # print(sum([len(i) for i in context]) / len(context))
    # raise KeyboardInterrupt

    json_path = '/shared_home/r12922051/Code/NTU-ADL-HW-2023Fall/HW1/output/lert_qa_plot/'
    with open(json_path+'metrics.json','r') as f:
        data = json.load(f)
    # print(data)

    for key in data:
        config = {
            'title': key,
            'xlabel': 'Epoch',
            'ylabel': key,
            'legend': True,
            'savefig': json_path+key+'.png'
        }
        config['data'] = {}
        if key == 'loss':
            config['data']['train'] = [list(range(len(data[key]))), [i[0] for i in data[key]]]
            config['data']['test'] = [list(range(len(data[key]))), [i[1] for i in data[key]]]
        else:
            config['data'][key] = [list(range(len(data[key]))), data[key]]

        print(config)
        myplot(config)