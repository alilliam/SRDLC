import matplotlib.pyplot as plt
import numpy as np
import os
import mpl_toolkits.mplot3d as p3d
from collections import Counter

# 查看准确率和损失函数
def plot_loss_2(d_loss,v_loss,num_epoch, epoches, save_dir):
    """
    Args:
        d_loss:A list of loss
        num_epoch:The current num of epoch
        epoches:The total number of epoch
        save_dir:The dir to save
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0,epoches + 1)
    #ax.set_ylim(0, max(np.max(v_loss), np.max(d_loss))* 1.1)
    ax.set_ylim(0, 1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    #plt.plot([i for i in range(1, num_epoch + 1)], d_loss, 'r', [i for i in range(1, num_epoch + 1)], v_loss, 'b')
    # plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='d_loss', color='mediumblue', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='v_acc', color='mediumblue', linewidth=2)
    plt.plot([i for i in range(1, num_epoch + 1)], v_loss, label='v_loss', color='red', linewidth=2)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_epoch_{}.png'.format(num_epoch)),dpi = 200)
    plt.close()


if __name__ == "__main__":
    pass