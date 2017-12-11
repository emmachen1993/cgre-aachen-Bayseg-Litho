import numpy as np
from sklearn import mixture
from scipy.stats import multivariate_normal, norm
from itertools import combinations
import tqdm  # smart-ish progress bar
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams  # plot arrangements

plt.style.use('bmh')



class litho:
    def __init__(litho, data, n_labels, segments):

        """

        :param data: Measurements
        :param n_label: label number
        :param clf.labels: segments


        """




indeces = np.where(data["Well Name"] == "SHRIMPLIN")[0]
test_well = data[data["Well Name"] == "SHRIMPLIN"]

label = clf.labels[-1][indeces]
print(label)
new_label = np.zeros(shape = (len(label),))
for i in range(0, len(label)):
    if i == 0:
        new_label[i] = 0
    elif label[i] == label[i-1]:
        new_label[i] = new_label[i-1]+0
    elif label[i] != label[i-1]:
        new_label[i] = new_label[i-1]+1
print(new_label)


L = np.zeros(shape=(len(np.unique(new_label)),))
for i in np.unique(new_label):
    index = np.where(new_label == i)
    pack = test_well.loc[index][feature_names].values
    den = np.zeros(shape = (len(pack),9))
    for l in range(len(pack)):
        for k in np.unique(data["Facies"]):
            litho_data = data[data["Facies"]==k][feature_names]
            den[l,k-1] = multivariate_normal.pdf(pack[l,:],np.mean(litho_data,axis=0),np.std(litho_data,axis=0))
    x = np.sum(den, axis=0).tolist().index(max(np.sum(den, axis=0)))
    L[int(i)] = int(x)
    L = L + 1
    print(L)


    def litho(L, label):
        count = 0
        new = np.zeros(shape=(len(label),))
        for i in range(0, len(label) - 1):
            if i == 0:
                new[i] = L[count]
            elif label[i] == label[i - 1]:
                new[i] = L[count]
            elif label[i] != label[i - 1]:
                count = count + 1
                new[i] = L[count]
        new[-1] = L[-1]
        return new

new = litho(L, new_label)
conpare = np.vstack((test_well["Facies"], new))
print(conpare)


sol = np.repeat(np.expand_dims(test_well["Facies"].values, 1), 100, 1)
ml_sol = np.repeat(np.expand_dims(new, 1), 100, 1)

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(2,13))

ax[0].imshow(sol, cmap="viridis")
ax[1].imshow(ml_sol, cmap="viridis")

np.count_nonzero(sol[:,1]-ml_sol[:,1])

ax[0].grid(False)
ax[1].grid(False)



