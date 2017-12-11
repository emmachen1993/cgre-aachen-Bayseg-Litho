def calc_sum_log_mixture_density_loop(self, comp_coef, mu, cov):
    """
    Calculate sum of log mixture density with each observation at every element.
    :param comp_coef: Component coefficient.
    :param mu: Mean matrix
    :param cov: Covariance matrix
    :return: summed log mixture density of the system
    """
    if self.dim == 1:
        lmd = 0.

        for x in range(len(self.coords)):
            storage2 = []
            for l in range(self.n_labels):
                a = comp_coef[x, l] * multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.obs[x])
                # print(a)
                storage2.append(a)

            lmd += (np.log(np.sum(storage2)))

    else:
        pass

    return lmd


def calc_gibbs_energy_loop(self, labels, beta):
    """
    Calculates Gibbs energy for each element using a penalty factor beta.
    :param labels: Array of labels at each element.
    :param beta: Energetic penalty parameter.
    :return: Gibbs energy for each element.
    """
    if self.dim == 1:
        # create ndarray for gibbs energy depending on element structure and n_labels
        gibbs_energy = np.zeros((len(self.coords), self.n_labels))
        for x, nl in enumerate(self.neighborhood):
            for n in nl:
                for l in range(self.n_labels):
                    if l != labels[n]:
                        gibbs_energy[x, l] += beta

    elif self.dim == 2:
        pass
    elif self.dim == 3:
        pass

    return gibbs_energy


def calc_energy_like_loop(self, mu, cov):
    """
    Calculates the energy likelihood of the system.
    :param mu: Mean values
    :param cov: Covariance matrix
    :return:
    """

    energy_like_labels = np.zeros((len(self.coords), self.n_labels))
    # print("energy like shp:", np.shape(energy_like_labels))

    for x in range(len(self.coords)):
        for l in range(self.n_labels):
            energy_like_labels[x, l] = 0.5 * np.array([self.obs[x] - mu[l, :]]) @ np.linalg.inv(cov[l, :, :]) @ np.array([self.obs[x] - mu[l, :]]).T + 0.5 * np.log(np.linalg.det(cov[l, :, :]))


    return energy_like_labels


def _define_neighborhood_system(coordinates):
    """

    :param coordinates:
    :return:
    """
    dim = np.shape(coordinates)[1]
    neighbors = [None for i in range(len(coordinates))]

    if dim == 1:
        for i, c in enumerate(coordinates):
            if i == 0:
                neighbors[i] = [i + 1]
            elif i == np.shape(coordinates)[0] - 1:
                neighbors[i] = [i - 1]
            else:
                neighbors[i] = [i - 1, i + 1]

    elif dim == 2:
        pass

    elif dim == 3:
        pass

    return neighbors

# l = labels.reshape(self.shape[:-1])
            # ge = np.tile(np.zeros_like(l).astype(float), (3, 1, 1))
            #
            # if self.stamp == 4:
            #     fp = np.array([[0, 1, 0],
            #                    [1, 0, 1],
            #                    [0, 1, 0]]).astype(bool)
            # else:
            #     fp = np.array([[1, 1, 1],
            #                    [1, 0, 1],
            #                    [1, 1, 1]]).astype(bool)
            #
            # for i in range(self.n_labels):
            #     ge[i, :, :] = generic_filter(l, partial(gibbs_comp_f, value=i), footprint=fp, mode="constant",
            #                                                cval=-999) * beta
            #
            # if verbose == "energy":
            #     print("\n")
            #     print("GIBBS ENERGY")
            #     print(ge)
            #
            # return ge.reshape(self.feat.shape[0], self.n_labels)



# # left column
            # # right
            # ge[:, :, 0] += np.not_equal(comp[:, :, 0], labels[:, 1]).astype(float) * beta
            # # above
            # ge[:, 1:, 0] += np.not_equal(comp[:, 1:, 0], labels[:-1, 1]).astype(float) * beta
            # # below
            # ge[:, :-1, 0] += np.not_equal(comp[:, :-1, 0], labels[1:, 1]).astype(float) * beta
            #
            # # right column
            # # left
            # ge[:, :, -1] += np.not_equal(comp[:, :, -1], labels[:, -2]).astype(float) * beta
            # # above
            # ge[:, 1:, -1] += np.not_equal(comp[:, 1:, -1], labels[:-1, -1]).astype(float) * beta
            # # below
            # ge[:, :-1, -1] += np.not_equal(comp[:, :-1, -1], labels[1:, -1]).astype(float) * beta
            #
            # # top row
            # # below
            # ge[:, 0, :] += np.not_equal(comp[:, 0, :], labels[1, :]).astype(float) * beta
            # # right
            # ge[:, 0, :-1] += np.not_equal(comp[:, 0, :-1], labels[0, 1:]).astype(float) * beta
            # # left
            # ge[:, 0, 1:] += np.not_equal(comp[:, 0, 1:], labels[0, :-1]).astype(float) * beta
            #
            # # bottom row
            # # above
            # ge[:, -1, :] += np.not_equal(comp[:, -1, :], labels[-2, :]).astype(float) * beta
            # # right
            # ge[:, -1, :-1] += np.not_equal(comp[:, -1, :-1], labels[-1, 1:]).astype(float) * beta
            # # left
            # ge[:, -1, 1:] += np.not_equal(comp[:, -1, 1:], labels[-1, :-1]).astype(float) * beta
            #
            # # corners redo
            # # up left
            # ge[:, 0, 0] = (np.not_equal(comp[:, 0, 0], labels[1, 0]).astype(float) + np.not_equal(comp[:, 0, 0], labels[0, 1]).astype(float)) * beta
            # # low left
            # ge[:, -1, 0] = (np.not_equal(comp[:, -1, 0], labels[-1, 1]).astype(float) + np.not_equal(comp[:, -1, 0], labels[-2, 0]).astype(float)) * beta
            # # up right
            # ge[:, 0, -1] = (np.not_equal(comp[:, 0, -1], labels[1, -1]).astype(float) + np.not_equal(comp[:, 0, -1], labels[0, -2]).astype(float)) * beta
            # # low right
            # ge[:, -1, -1] = (np.not_equal(comp[:, -1, -1], labels[-2, -1]).astype(float) + np.not_equal(comp[:, -1, -1], labels[-1, -2]).astype(float)) * beta
