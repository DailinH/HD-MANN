import numpy as np
import matplotlib.pyplot as plt
import torch
from cnn import CNNController
from data_generator import DataGenerator
from utils import * 
import sklearn
import sklearn.metrics

def inference(model, data_generator, device, key_mem_transform = binarize, n_step = 1000):
  model.eval()
  accumulated_acc = []
  for i in range(n_step):
    support_label, support_set, query_label, query_set = data_generator.sample_batch('val', 32)
    support_label, support_set = prep_data(support_label, support_set, device)
    query_label, query_set = prep_data(query_label, query_set, device)
    support_label = support_label.cpu().numpy()
    query_label = query_label.cpu().numpy()
    with torch.no_grad():
      support_keys = key_mem_transform(model(support_set).cpu().detach().numpy())
      query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
      dot_sim = get_dot_prod_similarity(query_keys, support_keys)
      sharpened = np.abs(dot_sim)
      pred = np.dot(sharpened, support_label)
      pred_argmax = np.argmax(pred, axis = 1)
      query_label_argmax = np.argmax(query_label, axis = 1)
      # print(np.sum(pred_argmax == query_label_argmax))
      accumulated_acc.append(np.sum(pred_argmax == query_label_argmax)/len(pred_argmax))
  return accumulated_acc

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

def plot_pairwise(sharpen='softabs'):
    W = 5
    S = 1
    D = 512
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    cosine_sim = get_pairwise_similarity('model_{}.pth'.format(sharpen), W, S, D, device)
    plt.figure()
    plt.pcolor(np.array(cosine_sim), cmap=plt.cm.jet, vmin=-1, vmax=1)#, cmap=plt.cm.seismic, vmin=0, vmax=2)
    plt.colorbar()
    plt.title(sharpen)
    # plt.imshow(cosine_sim, cmap='hot')
    plt.savefig('{}_pairwise.png'.format(sharpen))
# plot_loss_comparison()
# plot_acc_comparison()

def plot_bipolar_binary(sharpen='softabs'):
    W = 5
    S = 1
    D = 512

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = CNNController(D).float().to(device)
    model.load_state_dict(torch.load('model_softabs.pth'))
    model.eval()

    # cosine_sim = get_pairwise_similarity('model_{}.pth'.format(sharpen), W, S, D, device)
    data_gen = DataGenerator(W,S)
    acc_binarize = inference(model, data_gen, device, key_mem_transform=binarize, n_step=1000)
    acc_bipolarize = inference(model, data_gen, device, key_mem_transform=bipolarize, n_step=1000)
    data = np.asarray(acc_binarize), np.asarray(acc_bipolarize)
    # N_points = 100000
    n_bins = 20
    # Generate a normal distribution, center at x=0 and y=5
    x, y = data
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(x, bins=n_bins)
    axs[0].set_title('Binary')
    axs[0].set_xlabel('Classification Accuracy')
    axs[0].set_xlim([0, 1])

    axs[1].hist(y, bins=n_bins)
    axs[1].set_title('Bipolar')
    axs[1].set_xlabel('Classification Accuracy')
    axs[1].set_xlim([0, 1])

    fig.suptitle("Binary vs Bipolar in Histogram")
    plt.savefig('bipolar_binary_xcorr.png')

plot_bipolar_binary()