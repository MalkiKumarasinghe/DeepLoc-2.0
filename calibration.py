import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def draw_calibration_plot(model_attrs,y_test,y_test_preds)
    # Plot the Calibration Curve for every class
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    targets = range(len(model_attrs.classes_))
    for target in targets:
        prob_pos = y_test_preds
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test[:, target], y_test_preds, n_bins=10)
        name = model_attrs_[target]

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name, ))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("The proportion of samples whose class is the positive class")
    ax1.set_xlabel("The mean predicted probability")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plot for Deeploc 2.0 model predictions')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
