def bootstrap(data, n, axis=0, func=np.var, func_kwargs={"ddof": 1}):
    """Produce n bootstrap samples of data of the statistic given by func.

    Arguments
    ---------
    data : numpy.ndarray
        Data to resample.
    n : int
        Number of bootstrap trails.
    axis : int, optional
        Axis along which to resample. (Default ``0``).
    func : callable, optional
        Statistic to calculate. (Default ``numpy.var``).
    func_kwargs : dict, optional
        Dictionary with extra arguments for func. (Default ``{"ddof" : 1}``).

    Returns
    -------
    samples : numpy.ndarray
        Bootstrap samples of statistic func on the data.
    """

    fiducial_output = func(data, axis=axis, **func_kwargs)

    if isinstance(data, list):
        if axis != 0:
            raise NotImplementedError("Only axis == 0 supported.")
        assert all([d.shape[1:] == data[0].shape[1:] for d in data])

    samples = np.zeros((n, *fiducial_output.shape), dtype=fiducial_output.dtype)

    for i in range(n):
        print(i / n * 100, end="\r")
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True) for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)], axis=axis, **func_kwargs)
        else:
            axes = np.arange(len(data.shape))
            indices = (1, Ellipsis, 1)
            idx = np.random.choice(data.shape[axis], size=data.shape[axis], replace=True)
            idx_array = tuple(idx if ax == axis else Ellipsis for ax in axes)
            # [np.arange(data.shape[ax]) if ax != axis else idx for ax in axes]
            samples[i] = func(data[idx_array], axis=axis, **func_kwargs)
    print()
    return samples


def bootstrap_statistic_2d(data, binedges=None,axis=0):
        x,y = data
        f = np.histogram2d(
            x,y,
            bins=binedges,
            density=True,
        )
        return f[0].T





def get_stats_from_sims(sims, orders=[1, 2, 3], axis=1):

    dims = np.arange(sims.shape[axis - 1])
    n_sims = sims.shape[axis]
    print("number of simulations: " + str(n_sims))

    # np.cov needs to have the data in the form of (n, m) where n is the number of variables and m is the number of samples
    def moments_nd(order, sims):
        if order == 1:
            return np.mean(sims, axis=axis)
        elif order == 2:
            sims = sims - np.mean(sims, axis=axis)[:,None]
            if axis == 0:
                sims = sims.T
            
            cov = sims @ sims.T.conj() / n_sims
            np_cov = np.cov(sims, ddof=1)
            assert np.allclose(cov, np_cov), np_cov
            return cov

        else:

            combs = np.array(list(itertools.combinations_with_replacement(dims, order)))
            sims = sims - np.mean(sims, axis=axis)
            if axis == 0:
                sims = sims.T
            higher_moments = np.mean(np.prod(sims[combs], axis=1), axis=1)
            return np.ravel(higher_moments)

    stats = [moments_nd(order, sims) for order in orders]
    if len(orders) == 1:
        return np.array(stats)
    else:
        return stats


# In statistics.py
def compare_cf_to_simulations(t_grid, theory_cf, sim_data, n_moments=3):
    """Compare theoretical CF moments to simulation moments."""
    import moments
    
    theory_moments = moments.nth_moment(n_moments, t_grid, theory_cf)
    sim_moments = compute_simulation_moments(sim_data, orders=list(range(1, n_moments+1)))
    
    return {
        'theory': [m.real for m in theory_moments],
        'simulation': sim_moments,
        'difference': [t.real - s for t, s in zip(theory_moments, sim_moments)]
    }