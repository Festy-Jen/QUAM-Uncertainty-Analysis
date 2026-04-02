import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import matplotlib.pyplot as plt

def calculate_uncertainty_setting_b(
    average_net_pred: torch.Tensor,
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''

    :param average_net_pred:
    [n_points, n_categories]
    :param sample_preds:
    [n_points, n_posterior_samples, n_models, n_classes]
    :param sample_weights:
    [n_points, n_posterior_samples, n_models]
    :param kwargs:
    :return:
    '''

    if sample_weights is None:
        total = - torch.mean(
            torch.sum(
                (average_net_pred.unsqueeze(1).unsqueeze(2) * torch.log(sample_preds + gamma)), dim=-1), # .unsqueeze(2)
            dim=(1, 2))
        aleatoric = - torch.sum( (average_net_pred+gamma) * torch.log(average_net_pred+gamma), dim=-1)
        # print(total, aleatoric)
        epistemic = total - aleatoric
    else:
        # sample weight implied to be normalized
        total = - torch.sum(
            torch.sum(
                (average_net_pred.unsqueeze(1).unsqueeze(2) * torch.log(sample_preds + gamma)), dim=-1)*sample_weights, # .unsqueeze(2)
            dim=(1, 2))
        aleatoric = - torch.sum( (average_net_pred+gamma) * torch.log(average_net_pred+gamma), dim=-1)
        # print(total, aleatoric)
        epistemic = total - aleatoric

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }



def calculate_uncertainty_setting_a(
    average_net_pred: torch.Tensor,
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''
    if provided, uses the optimization steps
    :param kwargs:
    :return:
    '''
    if sample_weights is None:
        total = - torch.sum(torch.mean(sample_preds, dim=(0, 1)) * torch.log(torch.mean(sample_preds, dim=(0, 1))), dim=-1)
        aleatoric = - torch.mean(torch.sum(sample_preds * torch.log(sample_preds), dim=-1), dim=(0, 1))
        epistemic = total - aleatoric
    else:
        total = - torch.sum(torch.sum(sample_preds*sample_weights, dim=(0, 1)) * torch.log(torch.sum(sample_preds*sample_weights, dim=(0, 1))), dim=-1)
        aleatoric = - torch.mean(torch.sum(sample_preds * torch.log(sample_preds), dim=-1), dim=(0, 1))
        epistemic = total - aleatoric

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }

def evaluate_score(y, score):
    if score.dtype == torch.bfloat16:
        score = score.to(torch.float32)

    precision, recall, _ = precision_recall_curve(y, score)
    return {
        'FPR': float(fpr_at_tpr_x(y, score)),
        'AUROC': float(roc_auc_score(y, score)),
        'AUPR': float(auc(recall, precision)),
    }



def fpr_at_tpr_x(y_true, score, x=0.95):
    fpr, tpr, _ = roc_curve(y_true, score)
    return fpr[(np.abs(tpr - x)).argmin()]


#retention
def evaluate_missclass(
    id_uncerts,
    id_uncerts_target,
    id_uncerts_average_net_pred,
):
    # compute the "accuracy score" - 1 if correct, 0 if incorrect
    #print(id_uncerts_average_net_pred, id_uncerts_target)
    accuracy_scores = torch.argmax(id_uncerts_average_net_pred, dim=-1) != id_uncerts_target # [n_points]
    # the correct predictions - class 0, should be associated with lower uncertainty values\
    # the incorrect - class 1 with higher uncertainties
    return evaluate_score(accuracy_scores, id_uncerts.float())



def plot_uncertainty_correlation(
        aleatoric, epistemic, 
        title, filename,
        x_lims=(1e-13, 1e1), y_lims=(1e-14, 1e0)
):
    print(f"Generating {title} Plot...")

    fig, ax = plt.subplots(figsize=(8,8), dpi=100)

    hb = ax.hexbin(
        aleatoric, epistemic,
        gridsize=100,
        cmap='plasma',
        bins='log',
        xscale='log',yscale='log',
        mincnt=1,
        edgecolors='none'
    )

    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)

    ax.set_xlabel('Aleatoric Uncertainty', color='royalblue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epistemic Uncertainty', color='firebrick', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # colorbar with the temperature at the side
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Log10', fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{filename}", bbox_inches='tight', dpi=300)
    print(f"Success! Saved as {filename}")
    plt.show()
