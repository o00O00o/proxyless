import matplotlib.pyplot as plt

epoch_list = [50, 100, 150, 200]
for i, save_epoch in enumerate(epoch_list):
    path = '/home/gaoyibo/codes/proxyless/record/search_baseline/learned_net_{}/logs/valid_console.txt'.format(save_epoch)
    acc_list = []
    loss_list = []

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.split()
            if cols[0] == 'Valid':
                loss = float(cols[3])
                acc = float(cols[6])
                acc_list.append(acc)
                loss_list.append(loss)

    plt.plot(range(len(acc_list)), acc_list, label=str(save_epoch))

plt.legend()
plt.show()
