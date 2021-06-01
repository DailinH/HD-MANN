import numpy as np
import matplotlib.pyplot as plt

def plot_loss_comparison():
    plt.figure()
    arr_abs = np.load('softabs_data.npz')
    step_abs, loss_abs, val_abs = arr_abs['arr_0'], arr_abs['arr_1'], arr_abs['arr_2']
    arr_max = np.load('softmax_data.npz')
    step_max, loss_max, val_max = arr_max['arr_0'], arr_max['arr_1'], arr_max['arr_2']
    print(step_abs)
    plt.plot(step_abs, loss_abs, label='softabs')
    plt.plot(step_max, loss_max, label='softmax')
    plt.xlabel("iteration")
    plt.ylabel('training loss')
    plt.title("Training Loss")
    plt.legend()
    plt.savefig('training_loss.png')


def plot_acc_comparison():
    plt.figure()
    arr_abs = np.load('softabs_data.npz')
    step_abs, loss_abs, val_abs = arr_abs['arr_0'], arr_abs['arr_1'], arr_abs['arr_2']
    arr_max = np.load('softmax_data.npz')
    step_max, loss_max, val_max = arr_max['arr_0'], arr_max['arr_1'], arr_max['arr_2']
    print(step_abs)
    plt.plot(step_abs, val_abs, label='softabs')
    plt.plot(step_max, val_max, label='softmax')
    plt.xlabel("iteration")
    plt.ylabel('training accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig('training_accuracy.png')

# plot_loss_comparison()
# plot_acc_comparison()