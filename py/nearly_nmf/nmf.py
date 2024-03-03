#!/usr/bin/env python
import numpy as np
import numpy.typing as npt

nan_eps = 1e-6

# check if cupy is installed and a GPU is available
try:
    import cupy as cp
    if not cp.is_available():
        cp = None
except ImportError:
    cp = None

def _get_array_module(data, use_gpu=None):
    """
    Return either numpy or cupy depending upon type of `data` and `use_gpu`

    Parameters
    ----------
    data : array_like
        Input array to derive CPU or GPU usage
    use_gpu : bool, optional
        Override whether to use GPU or not; by default auto-derive

    Returns
    -------
    xp : module
        either cupy or numpy

    Raises
    ------
    ValueError if data is neither numpy nor cupy array
    """
    if use_gpu is None:
        if isinstance(data, np.ndarray):
            return np
        elif cp is not None and isinstance(data, cp.ndarray):
            return cp
        else:
            raise ValueError(f'Unknown array type {type(data)}')
    elif use_gpu:
        if cp is not None:
            return cp
        else:
            raise ValueError('use_gpu=True but GPU not available')
    else:
        return np

def shift_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike,
            W_start: npt.ArrayLike, n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, return_chi_2: bool = False,
            verbose: bool = False, transpose: bool = False) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the "shift-NMF" algorithm.

    WARNING: This function will do no sanity checking of inputs. It is highly recommended to use `fit_NMF`
    to fit templates instead of calling this function directly.

    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like
        Starting point for the H matrix. Must have shape (n_templates, n_observations).
    W_start : array_like
        Starting point for the W matrix. Must have shape (n_dimensions, n_templates).
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.
    verbose : bool, optional
        Whether to verbosely print the chi^2 values when tracking. This
        does nothing if return_chi_2 is False. Defaults to False.
    transpose : bool, optional
        Whether or not to train in "transposed" form, where the observations
        are the rows of X rather than the columns. Note that this
        flips the dimensions for all input arrays. Defaults to False.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """
    # GPU (cupy) or CPU (numpy)?
    xp = _get_array_module(X)

    # Copy H and W to avoid mutating the inputs
    # If transposed we'll just transpose it, train, and then return transposed
    # back to the input direction
    if transpose:
        X, V = xp.asarray(X).T, xp.asarray(V).T
        H, W = xp.array(H_start, copy=True).T, xp.array(W_start, copy=True).T
    else:
        X, V = xp.asarray(X), xp.asarray(V)
        H, W = xp.array(H_start, copy=True), xp.array(W_start, copy=True)

    chi_2 = []
    # Only shift if the lowest value of X is negative, otherwise
    # we can ignore the shifting
    shift = xp.min(X)
    if shift < 0:
        # Since the shift is negative we need to subtract it here rather
        # than add it to shift X so the minimum is 0
        X = X - shift
    else:
        shift = 0

    # The initial chi^2 pre fitting
    if return_chi_2:
        recon = W @ H
        c2 = xp.sum((xp.sqrt(V) * (X - (recon - shift))) ** 2)
        chi_2.append(c2)

    V_X = V * X # Weighted X, outside the loop for efficiency
    for i in range(n_iter):
        # H Step
        if update_H:
            H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H - shift)))

            # Here to ensure that nans are converted to zeros.
            # If the weights are set to 0 we might end up with
            # a division by 0 and a corresponding nan/inf that needs
            # to be handled correctly
            H = xp.nan_to_num(H, nan=nan_eps, posinf=nan_eps)

        # W Step
        if update_W:
            W = W * ((V_X) @ H.T) / ((V * (W @ H - shift)) @ H.T)
            W = xp.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        if return_chi_2:
            recon = W @ H
            c2 = xp.sum((xp.sqrt(V) * (X - (recon - shift))) ** 2)
            chi_2.append(c2)
            if verbose & (i % 10 == 0):
                print(i, c2)

    # Transposing back for returns
    if transpose:
        H = H.T
        W = W.T

    if return_chi_2:
        return H, W, chi_2
    else:
        return H, W

def split_pos_neg(A: npt.ArrayLike):
    """Splits a matrix into its positive and negative elements, with all other values set to 0.

    Parameters
    ----------
    A : array_like
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as A, with negative or zero elements set to 0.
    numpy.ndarray
        Array of the same shape as A, with positive or zero elements set to 0.
    """
    # GPU (cupy) or CPU (numpy)?
    xp = _get_array_module(A)

    A = xp.asarray(A)
    return (xp.abs(A) + A) / 2, (xp.abs(A) - A) / 2


def nearly_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike,
            W_start: npt.ArrayLike, n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, return_chi_2: bool = False, atol: float = 1e-5,
            verbose: bool = False, transpose: bool = False) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the "nearly-NMF" algorithm.

    WARNING: This function will do no sanity checking of inputs. It is highly recommended to use `fit_NMF`
    to fit templates instead of calling this function directly.
    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like
        Starting point for the H matrix. Must have shape (n_templates, n_observations).
    W_start : array_like
        Starting point for the W matrix. Must have shape (n_dimensions, n_templates).
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.
    verbose : bool, optional
        Whether to verbosely print the chi^2 values when tracking. This
        does nothing if return_chi_2 is False. Defaults to False.
    transpose : bool, optional
        Whether or not to train in "transposed" form, where the observations
        are the rows of X rather than the columns. Note that this
        flips the dimensions for all input arrays. Defaults to False.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """
    # GPU (cupy) or CPU (numpy)?
    xp = _get_array_module(X)

    # Copy H and W to avoid mutating the inputs
    # If transposed we'll just transpose it, train, and then return transposed
    # back to the input direction
    if transpose:
        X, V = xp.asarray(X).T, xp.asarray(V).T
        H, W = xp.array(H_start, copy=True).T, xp.array(W_start, copy=True).T
    else:
        X, V = xp.asarray(X), xp.asarray(V)
        H, W = xp.array(H_start, copy=True), xp.array(W_start, copy=True)

    chi_2 = []
    # The initial chi^2 pre fitting
    if return_chi_2:
        recon = W @ H
        c2 = xp.sum((xp.sqrt(V) * (X - recon)) ** 2)
        chi_2.append(c2)
    # Precomputing some values for efficiency
    _H, _W = None, None
    V_X = V * X
    for i in range(n_iter):
        # H-step
        if update_H:
            W_VX = W.T @ V_X
            W_VX_pos, W_VX_neg = split_pos_neg(W_VX)

            _H = H.copy()
            H = H * (W_VX_pos) / (W.T @ (V * (W @ H)) + W_VX_neg)
            H = xp.nan_to_num(H, nan=nan_eps, posinf=nan_eps)
            
        # W-step
        if update_W:
            V_XH = V_X @ H.T
            V_XH_pos, V_XH_neg = split_pos_neg(V_XH)

            _W = W.copy()
            W = W * (V_XH_pos) / ((V * (W @ H)) @ H.T + V_XH_neg)
            W = xp.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        if return_chi_2:
            recon = W @ H
            c2 = xp.sum((xp.sqrt(V) * (X - recon)) ** 2)
            chi_2.append(c2)
            if verbose & (i % 10 == 0):
                print(f'iteration {i} chi2={c2:.0f}')

        if _H is not None and _W is not None:
            if np.max(np.abs(H - _H)) < atol and np.max(np.abs(W - _W)) < atol:
                break

    # Transposing back for returns
    if transpose:
        H = H.T
        W = W.T

    if return_chi_2:
        return H, W, chi_2
    else:
        return H, W

def fit_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
            W_start: npt.ArrayLike = None, n_templates: int = 2,
            n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, algorithm: str = "nearly",
            return_chi_2: bool = False, verbose: bool = False,
            transpose: bool = False,
            use_gpu: bool = None,
            ) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the specified algorithm.

    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like, optional
        Starting point for the H matrix. Defaults to a matrix of random
        normal variables with a mean of 0 and a sigma of 2. If provided
        must have shape (n_templates, n_observations).
    W_start : array_like, optional
        Starting point for the W matrix. Defaults to a matrix of random
        normal variables with a mean of 0 and a sigma of 2. If provided
        must have shape (n_dimensions, n_templates).
    n_templates : int, optional
        Number of templates to fit. Not necessary when providing
        a starting matrix for H or W. Defaults to 2.
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    algorithm : {"shift", "nearly"}, optional
        The algorithm to use to do the fit. Defaults to "nearly".
            * "shift" : Uses the shift-NMF algorithm, where all data is shifted
            to the nonnegative half plane and NMF is fitted with a fixed
            template/coefficient pair accounting for the shift.
            * "nearly" : Uses the nearly-NMF algorithm, where the positive
            and negative components of the input data are separated and accounted
            for separately in the update rules.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.
    verbose : bool, optional
        Whether to verbosely print the chi^2 values when tracking. This
        does nothing if return_chi_2 is False. Defaults to False.
    transpose : bool, optional
        Whether or not to train in "transposed" form, where the observations
        are the rows of X rather than the columns. Note that this
        flips the dimensions for all input arrays. Defaults to False.
    use_gpu : bool, optional
        If True, use GPU even if input X and V are numpy arrays on CPU.
        By default (None), auto-derive whether to use GPU based upon
        numpy vs. cupy type of input X and V.
        use_gpu=False with cupy inputs is not supported.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """
    # GPU (cupy) or CPU (numpy)?
    xp = _get_array_module(X, use_gpu)

    # Move data to GPU if requested
    input_type = type(X)
    if use_gpu and input_type == np.ndarray:
        X = cp.asarray(X)
        V = cp.asarray(V)
        if H_start is not None:
            H_start = cp.asarray(H_start)
        if W_start is not None:
            W_start = cp.asarray(W_start)

    if use_gpu is False and cp is not None and input_type == cp.ndarray:
        raise TypeError('Input cupy arrays with use_gpu=False is not supported; move inputs to CPU first')

    if (H_start is not None) and (W_start is not None):
        if transpose:
            assert H_start.shape[1] == W_start.shape[0], "Number of templates does not match between H and W"
        else:
            assert H_start.shape[0] == W_start.shape[1], "Number of templates does not match between H and W"
    assert (update_H or update_W), "At least one of update_H or update_W must be True"

    if transpose:
        if H_start is not None: n_templates = H_start.shape[1]
        elif W_start is not None: n_templates = W_start.shape[0]
    else:
        if H_start is not None: n_templates = H_start.shape[0]
        elif W_start is not None: n_templates = W_start.shape[1]

    # Size of the coefficients and templates respectively
    # to ensure we use the same size everywhere
    if transpose:
        n_obs = X.shape[0]
        n_dim = X.shape[1]
    else:
        n_obs = X.shape[1]
        n_dim = X.shape[0]
    H_shape = (n_templates, n_obs)
    W_shape = (n_dim, n_templates)

    # numpy and cupy give different random numbers even with same seed,
    # so only use numpy for any random numbers, then move to GPU if needed
    rng = np.random.default_rng(100921)

    # Randomly initialize the H and W matrices if necessary.
    if H_start is not None:
        H = xp.asarray(H_start)
    else:
        H = xp.asarray(rng.uniform(0, 2, H_shape))
        # If we initialized, transpose to be of the "expected" form for
        # transpose.
        if transpose: H = H.T


    if W_start is not None:
        W = xp.asarray(W_start)
    else:
        W = xp.asarray(rng.uniform(0, 2, W_shape))
        if transpose: W = W.T

    if algorithm == "shift":
        to_return = shift_NMF(X, V, H, W, n_iter, update_H, update_W, return_chi_2, verbose, transpose)
    else:
        to_return = nearly_NMF(X, V, H, W, n_iter, update_H, update_W, return_chi_2, verbose, transpose)

    #- if forcing use_gpu, match return to input datatype
    if use_gpu is True and input_type == np.ndarray:
        to_return = tuple([tmp.get() for tmp in to_return])

    return to_return


class NMF:
    def __init__(self, X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
                 W_start: npt.ArrayLike = None, n_templates: int = 2, n_iter: int = 500,
                 algorithm: str = "nearly", return_chi_2: bool = False,
                 verbose: bool = False, transpose: bool = False, use_gpu: bool = None):
        """An NMF model object. This object holds all of the relevant NMF algorithmic data, namely
        fitted coefficients and templates/basis vectors for its training dataset.

        Parameters
        ----------
        X : array_like
            Input data of shape (n_dimensions, n_observations).
        V : array_like
            Input weights of shape (n_dimensions, n_observations).
        H_start : array_like, optional
            Starting point for the H matrix. Defaults to a matrix of random
            normal variables with a mean of 0 and a sigma of 2. If provided
            must have shape (n_templates, n_observations).
        W_start : array_like, optional
            Starting point for the W matrix. Defaults to a matrix of random
            normal variables with a mean of 0 and a sigma of 2. If provided
            must have shape (n_dimensions, n_templates).
        n_templates : int, optional
            Number of templates to fit. Not necessary when providing
            a starting matrix for H or W. Defaults to 2.
        n_iter : int, optional
            Number of fitting iterations to run. Defaults to 500.
        algorithm : {"shift", "nearly"}, optional
            The algorithm to use to do the fit. Defaults to "nearly".
            * "shift" : Uses the shift-NMF algorithm, where all data is shifted
            to the nonnegative half plane and NMF is fitted with a fixed
            template/coefficient pair accounting for the shift.
            * "nearly" : Uses the nearly-NMF algorithm, where the positive
            and negative components of the input data are separated and accounted
            for separately in the update rules.
        return_chi_2 : bool, optional
            Whether to track and return the chi^2 history of the fit. This
            involves computing a matrix norm, so will slightly slow the
            fit depending on data size. Defaults to False.
        verbose : bool, optional
            Whether to verbosely print the chi^2 values when tracking. This
            does nothing if return_chi_2 is False. Defaults to False.
        transpose : bool, optional
            Whether or not to train in "transposed" form, where the observations
            are the rows of X rather than the columns. Note that this
            flips the dimensions for all input arrays. Defaults to False.
        use_gpu : bool, optional
            Override whether to use GPU or not; by default auto-derive

        """
        # GPU (cupy) or CPU (numpy)?
        xp = _get_array_module(X)

        self.X, self.V = xp.asarray(X), xp.asarray(V)
        self.use_gpu = use_gpu

        if (H_start is not None) and (W_start is not None):
            if transpose:
                assert H_start.shape[1] == W_start.shape[0], "Number of templates does not match between H and W"
            else:
                assert H_start.shape[0] == W_start.shape[1], "Number of templates does not match between H and W"
        self.n_templates = n_templates
        if transpose:
            if H_start is not None: self.n_templates = H_start.shape[1]
            elif W_start is not None: self.n_templates = W_start.shape[0]
        else:
            if H_start is not None: self.n_templates = H_start.shape[0]
            elif W_start is not None: self.n_templates = W_start.shape[1]

        # Size of the coefficients and templates respectively
        # to ensure we use the same size everywhere
        if transpose:
            n_obs = X.shape[0]
            n_dim = X.shape[1]
        else:
            n_obs = X.shape[1]
            n_dim = X.shape[0]

        H_shape = (self.n_templates, n_obs)
        W_shape = (n_dim, self.n_templates)
        # numpy and cupy give different random numbers even with same seed,
        # so only use numpy for any random numbers, then move to GPU if needed
        self.rng = np.random.default_rng(100921)
        self.transpose = transpose

        # Randomly initialize the H and W matrices if necessary.
        if H_start is not None:
            self.H = xp.asarray(H_start)
        else:
            self.H = xp.asarray(self.rng.uniform(0, 2, H_shape))
            if transpose: self.H = self.H.T

        if W_start is not None:
            self.W = xp.asarray(W_start)
        else:
            self.W = xp.asarray(self.rng.uniform(0, 2, W_shape))
            if transpose: self.W = self.W.T

        # Options to pass to fit_NMF
        self.algorithm = algorithm
        self.return_chi_2 = return_chi_2
        self.verbose = verbose

        self.chi_2 = []
        self.n_iter = n_iter

    def fit(self):
        """Fit this NMF object to noisy, possibly negative, data with weights.
        """
        if self.return_chi_2:
            self.H, self.W, self.chi_2 = fit_NMF(self.X, self.V, self.H, self.W,
                                                      n_iter=self.n_iter, return_chi_2=self.return_chi_2,
                                                      verbose=self.verbose, transpose=self.transpose,
                                                      use_gpu=self.use_gpu, algorithm=self.algorithm)
        else:
            self.H, self.W = fit_NMF(self.X, self.V, self.H, self.W,
                                          n_iter=self.n_iter, transpose=self.transpose, use_gpu=self.use_gpu,
                                          algorithm=self.algorithm)

    def predict(self) ->  npt.NDArray:
        """Generate a reconstruction of the original input data using the stored factorization.

        Returns
        -------
         numpy.ndarray
            A reconstruction of the original input dataset, with same shape and
            size.
        """
        if self.transpose:
            return self.H @ self.W
        else:
            return self.W @ self.H

    def fit_coeffs(self, X: npt.ArrayLike, V: npt.ArrayLike) -> npt.NDArray:
        """Generate coefficients fitting this object's NMF templates
        to noisy, possibly negative, data with weights.

        Parameters
        ----------
        X : array_like
            Input data of shape (n_dimensions, n_observations). Dimensions should
            be flipped if the object was instantiated with `tranpose=True`.
        V : array_like
            Input weights of shape (n_dimensions, n_observations). Dimensions should
            be flipped if the object was instantiated with `tranpose=True`.

        Returns
        -------
        H : numpy.ndarray
            The fitted NMF coefficients with shape (n_templates,
            n_observations) for the objects internal templates.
            Each column represents the coefficients corresponding
            to a column of X.
        """
        return self.fit_NMF(X, V, self.H, self.W, n_iter=self.n_iter, update_W=False)[0]

