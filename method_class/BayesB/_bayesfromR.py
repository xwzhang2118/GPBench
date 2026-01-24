import numpy as np
import rpy2
from rpy2.robjects import numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import _param_free_base_model
from joblib import Parallel, delayed

class Bayes_R(_param_free_base_model.ParamFreeBaseModel):
    """
    Implementation of a class for Bayesian alphabet.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._param_free_base_model.ParamFreeBaseModel` for more information on the attributes.

        *Additional attributes*

        - mu (*np.array*): intercept
        - beta (*np.array*): effect size
        - model_name (*str*): model to use (BayesA, BayesB or BayesC)
        - n_iter (*int*): iterations for sampling
        - burn_in (*int*): warmup/burnin for sampling
    """
    standard_encoding = '012'
    possible_encodings = ['101']

    def __init__(self, task: str, model_name: str, encoding: str = None, n_iter: int =1000, burn_in: int = 200):
        super().__init__(task=task, encoding=encoding)
        self.model_name = model_name
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n_jobs = 1
        self.mu = None
        self.beta = None

    def _run_chain(self, chain_num: int, R_X, R_y):
        """
        Helper function to run an individual MCMC chain.
        """
        BGLR = importr('BGLR')

        # Run BGLR for BayesB on a single chain
        ETA = robjects.r['list'](robjects.r['list'](X=R_X, model=self.model_name))
        fmBB = BGLR.BGLR(y=R_y, ETA=ETA, verbose=False, nIter=self.n_iter, burnIn=self.burn_in)

        # Extract the results for this chain
        beta_chain = np.asarray(fmBB.rx2('ETA').rx2(1).rx2('b'))
        mu_chain = np.asarray(fmBB.rx2('mu'))  # Extract mu (intercept) for this chain
        return beta_chain, mu_chain

    def fit(self, X: np.array, y: np.array) -> np.array:
        """
        Implementation of fit function for Bayesian alphabet imported from R.

        See :obj:`~easypheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        # import necessary R packages
        base = importr('base')
        BGLR = importr('BGLR')

        # create R objects for X and y
        R_X = robjects.r['matrix'](X, nrow=X.shape[0], ncol=X.shape[1])
        R_y = robjects.FloatVector(y)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_chain)(chain_num, R_X, R_y) for chain_num in range(self.n_jobs)
        )

        # Aggregate results from all chains
        beta_chains = [result[0] for result in results]
        mu_chains = [result[1] for result in results]

        # Compute the mean of beta and mu over all chains
        self.beta = np.mean(beta_chains, axis=0)
        self.mu = np.mean(mu_chains, axis=0)

        # run BGLR for BayesB
        # ETA = base.list(base.list(X=R_X, model=self.model_name))
        # fmBB = BGLR.BGLR(y=R_y, ETA=ETA, verbose=True, nIter=self.n_iter, burnIn=self.burn_in)

        # # save results as numpy arrays
        # self.beta = np.asarray(fmBB.rx2('ETA').rx2(1).rx2('b'))
        # self.mu = fmBB.rx2('mu')
        return self.predict(X_in=X)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of predict function for Bayesian alphabet model imported from R.

        See :obj:`~easypheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return self.mu + np.matmul(X_in, self.beta)
