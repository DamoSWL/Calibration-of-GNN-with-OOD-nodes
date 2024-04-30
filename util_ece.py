import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn



class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.exp(logits)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece




def convert2one_hot(labels, nclass, device):
    labels = labels.tolist()
    one_hot = []
    for l in range(len(labels)):
        cur_sample = []
        for i in range(nclass):
            if i == labels[l]:
                cur_sample.append(1)
            else:
                cur_sample.append(0)
        one_hot.append(cur_sample)
    return torch.Tensor(one_hot).to(device)

def calc_bins(labels_oneh, preds):
  # Assign each prediction to a bin
  num_bins = 10 
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)


  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]
  # print(f'bins {bins} binned {binned} bin_acc {bin_accs} bin_confs {bin_confs} bin_sizes {bin_sizes}')
  return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(labels_oneh, preds):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels_oneh, preds)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)
  return ECE, MCE


def draw_reliability_graph(labels_oneh, preds, dataset_string, model_string, task_type,seed):
  ECE, MCE = get_metrics(labels_oneh, preds)
  bins, _, bin_accs, _, _ = calc_bins(labels_oneh, preds)

  fig = plt.figure(figsize=(6, 4))
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams["font.weight"] = "bold"
  
  ax = fig.gca()

  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  plt.xlabel('Confidence',fontsize=14)
  plt.ylabel('Accuracy',fontsize=14)
  ax.set_axisbelow(True) 

  # Error bars
  plt.bar(bins, bins, width=0.09, alpha=0.7, color='lightcoral', label='Expected') 

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.09, alpha=0.7, color='dodgerblue', label='Outputs') 
  plt.plot([0,1],[0,1], '--', c='k', linewidth=1)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box') #add to get square

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.tick_params(labelsize=13)
  plt.legend(fontsize=14)
  today_date = datetime.today().strftime('%Y-%m-%d')
  plt.savefig(str(seed)+ '_' + today_date + '_ECE_plot_'+ model_string + '_' + dataset_string + '_' + task_type +'.png', bbox_inches='tight',format='png', dpi=300,
                pad_inches=0)
  
  return ECE
