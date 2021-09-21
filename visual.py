import matplotlib.pyplot as plt


epoch_list = [50, 100, 150, 200, 250, 300, 350, 400, 450]
test_acc_list = []
for i, save_epoch in enumerate(epoch_list):
    path = '/home/gaoyibo/codes/proxyless/record/search_baseline/learned_net_{}/logs/valid_console.txt'.format(save_epoch)
    output_path = '/home/gaoyibo/codes/proxyless/record/search_baseline/learned_net_{}/output'.format(save_epoch)
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
    
    plt.subplot(121)
    plt.plot(range(len(acc_list)), acc_list, label=str(save_epoch))
    plt.legend()

    with open(output_path, 'r') as f:
        lines = f.readlines()
        test_acc_list.append(float(lines[2].split('\"')[3]))


print(test_acc_list)
plt.subplot(122)
plt.bar(range(len(test_acc_list)), test_acc_list, tick_label=epoch_list)

plt.show()
