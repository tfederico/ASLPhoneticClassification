import numpy as np
from sklearn.metrics import auc, plot_roc_curve, plot_precision_recall_curve, precision_recall_curve


def plot_mean_curve(y, mean_x, init_val, aus, ax):
    mean_y = np.mean(y, axis=0)
    if init_val != None:
        mean_y[-1] = init_val
    mean_area = auc(mean_x, mean_y)
    std_area = np.std(aus)
    if init_val != None:
        ax.plot(mean_x, mean_y, color='b',
                label=r'Mean (%0.2f $\pm$ %0.2f)' % (mean_area, std_area),
                lw=2, alpha=.8)
    else:
        ax.plot(mean_y, mean_x, color='b',
                label=r'Mean (%0.2f $\pm$ %0.2f)' % (mean_area, std_area),
                lw=2, alpha=.8)

    return mean_y

def plot_std_curve(y, mean_y, mean_x, ax):
    std = np.std(y, axis=0)
    upper = np.minimum(mean_y + std, 1)
    lower = np.maximum(mean_y - std, 0)
    ax.fill_between(mean_x, lower, upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

def plot_random_chance_line(bounds_x, bounds_y, ax):
    ax.plot(bounds_x, bounds_y, linestyle='--', lw=2, color='r',
            label='Random', alpha=.8)

def set_axes_lim_legend_pos(xlim, ylim, title, loc, ax):
    ax.set(xlim=xlim, ylim=ylim, title=title)
    ax.legend(loc=loc)


def plot_curves():
    pass
#     viz0 = plot_roc_curve(classifier, X[val], y[val],
#                          name='ROC fold {}'.format(i),
#                          alpha=0.5, lw=1, ax=ax[0])
#     viz1 = plot_precision_recall_curve(classifier, X[val], y[val],
#                          name='PR fold {}'.format(i),
#                          alpha=0.5, lw=1, ax=ax[1])
#     interp_tpr = np.interp(mean_fpr, viz0.fpr, viz0.tpr)
#     interp_prec = np.interp(mean_recall, viz1.precision, viz1.recall)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     precs.append(interp_prec)
#     aucs.append(viz0.roc_auc)
#     auprs.append(viz1.average_precision)
#
# mean_tpr = plot_mean_curve(tprs, mean_fpr, 1.0, aucs, ax[0])
# mean_prec = plot_mean_curve(precs, mean_recall, None, auprs, ax[1])
#
# plot_std_curve(tprs, mean_tpr, mean_fpr, ax[0])
# plot_std_curve(precs, mean_prec, mean_recall, ax[1])
#
# plot_random_chance_line([0, 1], [0, 1], ax[0])
# plot_roc_curve(classifier, test_features, test_labels, name='ROC test',
#                 alpha=0.5, lw=1, ax=ax[0])
# plot_random_chance_line([0, 1], [1, 0], ax[1])
# plot_precision_recall_curve(classifier, test_features, test_labels, name='PR test',
#                 alpha=0.5, lw=1, ax=ax[1])
#
# set_axes_lim_legend_pos([-0.05, 1.05], [-0.05, 1.05],
#                         "Receiver operating characteristic", "lower right", ax[0])
# set_axes_lim_legend_pos([-0.05, 1.05], [-0.05, 1.05],
#                         "Precision-Recall curve", "upper right", ax[1])
