import matplotlib.pyplot as plt

full_v2_path = '/home/gaoyibo/codes/proxyless/record/full_v2/logs/valid_console.txt'
two_path = '/home/gaoyibo/codes/proxyless/record/two/logs/valid_console.txt'
full_v2_acc_list = []
two_acc_list = []
full_v2_loss_list = []
two_loss_list = []

with open(full_v2_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        cols = line.split()
        if cols[0] == 'Valid':
            loss = float(cols[3])
            acc = float(cols[6])
            full_v2_acc_list.append(acc)
            full_v2_loss_list.append(loss)

with open(two_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        cols = line.split()
        if cols[0] == 'Valid':
            loss = float(cols[3])
            acc = float(cols[6])
            two_acc_list.append(acc)
            two_loss_list.append(loss)

plt.subplot(121)
plt.plot(range(len(full_v2_loss_list)), full_v2_loss_list, label='full_v2')
plt.plot(range(len(two_loss_list)), two_loss_list, label='two')
plt.legend()

plt.subplot(122)
plt.plot(range(len(full_v2_acc_list)), full_v2_acc_list, label='full_v2')
plt.plot(range(len(two_acc_list)), two_acc_list, label='two')
plt.legend()

plt.show()
