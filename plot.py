import numpy as np
import matplotlib.pyplot as plt
import torch
from cnn import CNNController
from data_generator import DataGenerator
from utils import * 
import sklearn
import sklearn.metrics

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

def get_pairwise_similarity(model_path, W, S, D, device):
    model = CNNController(D).float().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    accumulated_acc = 0
    data_gen = DataGenerator(W,S)
    support_label, support_set, query_label, query_set = data_gen.sample_batch('val', 1, False)

    support_label, support_set = prep_data(support_label, support_set, device)
    support_label = support_label.cpu().numpy()

    with torch.no_grad():
        support_keys = model(support_set).cpu().detach().numpy()
        dot_sim = sklearn.metrics.pairwise.cosine_similarity(support_keys, support_keys)
    return dot_sim

def plot_pairwise():
    W = 5
    S = 1
    D = 512
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    cosine_sim = get_pairwise_similarity('model_softmax.pth', W, S, D, device)
    plt.figure()
    plt.pcolor(np.array(cosine_sim), cmap=plt.cm.jet)#, cmap=plt.cm.seismic, vmin=0, vmax=2)
    plt.colorbar()
    # plt.imshow(cosine_sim, cmap='hot')
    plt.savefig('softmax_pairwise.png')
# plot_loss_comparison()
# plot_acc_comparison()

plot_pairwise()