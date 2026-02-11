
def _compute_variances(self, auto_prods, cross_prods, cross_combs):
    # eventually remove and use get_covariance_lowell
    logger.info("Computing variances...")
    auto_transposes = jnp.transpose(auto_prods, (0, 1, 3, 2))
    
    # Handle doubled data vector when xi_minus is included
    # should include large angle distinction in initialization, but actually, variances are only used in cf compuation, so only large angles here is fine.
    
    variances = jnp.zeros(self.data_shape_full)

    # Auto terms
    variances = variances.at[~self._is_cov_cross].set(
        2 * jnp.sum(auto_prods * auto_transposes, axis=(-2, -1))
    )

    # Cross terms
    cross_transposes = jnp.transpose(cross_prods, (0, 1, 3, 2))
    auto_normal, auto_transposed = cross_combs[:, 0], cross_combs[:, 1]
    variances = variances.at[self._is_cov_cross].set(jnp.sum(
        cross_prods * cross_transposes, axis=(-2, -1)
    ) + jnp.sum(
        auto_prods[auto_normal] * auto_transposes[auto_transposed], axis=(-2, -1))
    )
    logger.info("Variances computed.")
    if self.config.enable_memory_cleanup:
        del auto_transposes, cross_transposes, auto_prods, cross_prods
    return np.asarray(variances)