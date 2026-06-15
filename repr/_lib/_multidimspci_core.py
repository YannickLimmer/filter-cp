# Vendored from hamrel-cxu/MultiDimSPCI @ 2b22e47
# Commit: 2b22e47088ed37ebc48d1bb9fdfa192450f289a2
# Fetch date: 2026-04-25
# Patch: `from helpers import utils_SPCI as utils` → `from . import _multidimspci_utils as utils`

import math
import time as time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors

from . import _multidimspci_utils as utils

warnings.filterwarnings("ignore")


class SPCI_and_EnbPI:
    def __init__(self, X_train, X_predict, Y_train, Y_predict, fit_func):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        n, n1 = len(self.X_train), len(self.X_predict)
        self.d = self.Y_train.shape[1]  # dimension of output
        self.Ensemble_train_interval_centers = np.ones((n, self.d)) * np.inf
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = np.ones((n1, self.d)) * np.inf
        self.Ensemble_online_resid = np.ones((n + n1, self.d)) * np.inf  # LOO scores
        self.beta_hat_bins = []
        self.cov_matrix_ls = []
        #### Other hyperparameters for training
        # QRF training & how it treats the samples
        self.weigh_residuals = False  # Whether we weigh current residuals more.
        self.c = 0.995  # If self.weight_residuals, weights[s] = self.c ** s, s>=0
        self.n_estimators = 10  # Num trees for QRF
        self.max_d = 2  # Max depth for fitting QRF
        self.criterion = "squared_error"
        # search of \beta^* \in [0,\alpha]
        self.bins = 5  # break [0,\alpha] into bins to minimize width
        # how many LOO training residuals to use for training current QRF
        self.T1 = None  # None = use all
        # Extra: possible low-rank approximation
        self.r = None
        # Extra: local ellipsoid
        self.use_local_ellipsoid = False
        self.local_ellipsoid_idx = 0

    def one_boot_prediction(self, Xboot, Yboot, Xfull):
        model = self.regressor
        model.fit(Xboot, Yboot)
        return model.predict(Xfull)

    def fit_bootstrap_models_online_multistep(self, B, stride=1):
        """
        Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated
        predictors, and compute the residuals.

        stride: int. If > 1, then we perform multi-step prediction.
        """
        n = self.X_train.shape[0]
        n1 = len(self.X_predict)
        N = n - stride + 1
        train_pred_idx = np.arange(0, n, stride)
        test_pred_idx = np.arange(n, n + n1, stride)
        self.train_idx = train_pred_idx
        self.test_idx = test_pred_idx
        Xfull = np.vstack(
            [self.X_train[train_pred_idx], self.X_predict[test_pred_idx - n]]
        )
        nsub, n1sub = len(train_pred_idx), len(test_pred_idx)
        for s in range(stride):
            boot_samples_idx = utils.generate_bootstrap_samples(N, N, B)
            in_boot_sample = np.zeros((B, N), dtype=bool)
            boot_predictionsFX = np.zeros((B, nsub + n1sub, self.d))
            out_sample_predictFX = np.zeros((n, n1sub, self.d))

            start = time.time()
            for b in range(B):
                self.b = b
                Xboot, Yboot = (
                    self.X_train[boot_samples_idx[b], :],
                    self.Y_train[s : s + N][boot_samples_idx[b]],
                )
                in_boot_sample[b, boot_samples_idx[b]] = True
                boot_fX_pred = self.one_boot_prediction(Xboot, Yboot, Xfull)
                boot_predictionsFX[b] = boot_fX_pred
            print(
                f"{s+1}/{stride} multi-step: finish Fitting {B} Bootstrap models, "
                f"took {time.time()-start} secs."
            )

            start = time.time()
            for j, i in enumerate(train_pred_idx):
                if i < N:
                    b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        b_keep = 0
                else:
                    b_keep = range(B)
                pred_iFX = boot_predictionsFX[b_keep, j].mean(axis=0)
                pred_testFX = boot_predictionsFX[b_keep, nsub:].mean(axis=0)
                true_idx = min(i + s, n - 1)
                self.Ensemble_train_interval_centers[true_idx] = pred_iFX
                resid_LOO = self.Y_train[true_idx] - pred_iFX
                out_sample_predictFX[i] = pred_testFX
                self.Ensemble_online_resid[true_idx] = resid_LOO
            sorted_out_sample_predictFX = out_sample_predictFX[train_pred_idx].mean(0)
            pred_idx = np.minimum(test_pred_idx - n + s, n1 - 1)
            self.Ensemble_pred_interval_centers[pred_idx] = sorted_out_sample_predictFX
            pred_full_idx = np.minimum(test_pred_idx + s, n + n1 - 1)
            resid_out_sample = (
                self.Y_predict[pred_idx] - sorted_out_sample_predictFX
            )
            self.Ensemble_online_resid[pred_full_idx] = resid_out_sample
            print(
                f"Leave-one-out residuals computed, took {time.time()-start} secs."
            )
        num_inf = (self.Ensemble_online_resid == np.inf).sum()
        if num_inf > 0:
            print(
                f"Something can be wrong, as {num_inf}/{n+n1} residuals are not all computed"
            )
        self.get_test_et = False
        self.train_et = self.get_et(self.Ensemble_online_resid[:n])
        self.get_test_et = True
        self.test_et = self.get_et(self.Ensemble_online_resid[n:])
        self.all_et = np.concatenate([self.train_et, self.test_et])

    def get_local_ellipsoid(self):
        if self.use_local_ellipsoid and self.get_test_et:
            idx = self.local_ellipsoid_idx
            X_prev = np.vstack([self.X_train[idx:], self.X_predict[:idx]])
            max_past = min(1000, len(X_prev))
            X_prev = X_prev[-max_past:]
            n_neighbors = int(0.1 * max_past)
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_prev)
            neighbors = (
                knn.kneighbors(
                    self.X_predict[idx].reshape(1, -1), return_distance=False
                ).reshape(-1)
            )
            Cov_neighbor = np.cov(self.Ensemble_online_resid[idx:][neighbors].T)
            lamb = 0.95
            local_cov = lamb * Cov_neighbor + (1 - lamb) * self.global_cov
            cov_now, inv_cov_now = self.get_rank_approx(local_cov)
            self.cov_matrix_ls.append(cov_now)
            self.local_ellipsoid_idx += 1
            if self.local_ellipsoid_idx % 25 == 0:
                print(X_prev.shape)
                print(f"Local Ellipsoid {self.local_ellipsoid_idx} computed")
        return inv_cov_now

    def get_rank_approx(self, A):
        r = self.r
        if r is not None:
            u, s, v = np.linalg.svd(A, full_matrices=False)
            Ur = u[:, :r]
            Sr = np.diag(s[:r])
            Vr = v[:r, :]
            Ar = np.dot(Ur, np.dot(Sr, Vr))
            S_inv = np.diag(1 / s[:r])
            Ar_pseudo_inverse = np.dot(Vr.T, np.dot(S_inv, Ur.T))
        else:
            Ar = A
            Ar_pseudo_inverse = np.linalg.inv(A)
        return Ar, Ar_pseudo_inverse

    def get_et(self, residuals):
        """Compute Mahalanobis nonconformity scores from residuals."""
        if self.get_test_et is False:
            global_cov, global_inv = self.get_rank_approx(np.cov(residuals.T))
            self.global_cov = global_cov
            self.global_cov_inv = global_inv
        nonconform_scores = []
        for i in range(len(residuals)):
            if self.use_local_ellipsoid is False:
                cov_mat_est_inv = self.global_cov_inv
            else:
                if self.get_test_et is False:
                    cov_mat_est_inv = self.global_cov_inv
                else:
                    cov_mat_est_inv = self.get_local_ellipsoid()
            nonconform_scores.append(
                np.sqrt(
                    np.matmul(residuals[i], np.matmul(cov_mat_est_inv, residuals[i].T))
                )
            )
        return np.array(nonconform_scores)

    def compute_Widths_Ensemble_online(
        self,
        alpha,
        stride=1,
        smallT=True,
        past_window=100,
        use_SPCI=False,
        quantile_regr="RF",
    ):
        """Compute prediction interval widths.

        use_SPCI=False: uses empirical quantile (EnbPI mode, no sklearn_quantile needed).
        use_SPCI=True: uses quantile regression forest (requires sklearn_quantile).
        """
        self.alpha = alpha
        n1 = len(self.X_train)
        self.past_window = past_window
        if smallT:
            n1 = min(self.past_window, len(self.X_train))
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        if use_SPCI:
            s = stride
            stride = 1
        resid_strided = utils.strided_app(
            self.all_et[len(self.X_train) - n1 : -1], n1, stride
        )
        print(f"Shape of slided e_t lists is {resid_strided.shape}")
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        self.QRF_ls = []
        self.i_star_ls = []
        for i in range(num_unique_resid):
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                rfqr = self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all) / 2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid + i_star]
                width_left[i] = wid_left
                width_right[i] = wid_right
            else:
                past_resid = resid_strided[i, :]
                cov_mat = (
                    self.global_cov
                    if self.use_local_ellipsoid is False
                    else self.cov_matrix_ls[i]
                )
                beta_hat_bin = utils.binning(past_resid, cov_mat, alpha, self.bins)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = np.percentile(
                    past_resid, math.ceil(100 * beta_hat_bin)
                )
                width_right[i] = np.percentile(
                    past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin))
                )
            num_print = int(num_unique_resid / 20)
            if num_print == 0:
                print(f"Radius at test {i}: {width_right[i]-width_left[i]:.4f}")
            elif i % num_print == 0:
                print(f"Radius at test {i}: {width_right[i]-width_left[i]:.4f}")
        print(
            f"Finish Computing {num_unique_resid} unique Prediction Intervals, "
            f"took {time.time()-start} secs."
        )
        Ntest = len(out_sample_predict)
        width_left = np.repeat(width_left, stride)[:Ntest]
        width_right = np.repeat(width_right, stride)[:Ntest]
        Width_Ensemble = pd.DataFrame(
            np.c_[width_left, width_right], columns=["lower", "upper"]
        )
        self.Width_Ensemble = Width_Ensemble

    def get_results(self):
        covered_or_not, rolling_size = [], []
        for i in range(len(self.test_et)):
            et = self.test_et[i]
            lower = self.Width_Ensemble.iloc[i, 0]
            upper = self.Width_Ensemble.iloc[i, 1]
            covered_or_not.append((et <= upper) and (et >= lower))
            cov_mat = (
                self.global_cov
                if self.use_local_ellipsoid is False
                else self.cov_matrix_ls[i]
            )
            upper_v = utils.ellipsoid_volume(cov_mat, upper)
            lower_v = utils.ellipsoid_volume(cov_mat, lower)
            rolling_size.append(upper_v - lower_v)
        self.coverages_all = covered_or_not
        self.width_all = rolling_size
        mean_cov = np.mean(covered_or_not)
        mean_size = np.mean(rolling_size)
        print(
            f"Average Coverage is {mean_cov:.3f}, "
            f"Average Ellipsoid Volume is {mean_size:.2e}"
        )
        return mean_cov, mean_size

    def multi_step_QRF(self, past_resid, i, s, n2):
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[: num - s + 1], window_shape=n2)
        self.cov_matrix = (
            self.global_cov
            if self.use_local_ellipsoid is False
            else self.cov_matrix_ls[i]
        )
        for k in range(s):
            residY = past_resid[n2 + k : num - (s - k - 1)]
            self.train_QRF(residX, residY)
            if i == 0:
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        return resid_pred

    def train_QRF(self, residX, residY):
        try:
            from sklearn_quantile import (
                RandomForestQuantileRegressor,
                SampleRandomForestQuantileRegressor,
            )
        except ImportError as e:
            raise ImportError(
                "sklearn_quantile is required for SPCI mode. "
                "Install with: uv add sklearn-quantile. "
                "Use use_SPCI=False for the EnbPI empirical-quantile mode."
            ) from e
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        self.common_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_d,
            criterion=self.criterion,
            n_jobs=-1,
        )
        if residX[:-1].shape[0] > 10000:
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas
            )
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas
            )
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY))
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr,
                self.cov_matrix,
                residX[-(self.T1 + 1) : -1],
                residY[-self.T1 :],
                residX[-1],
                beta_ls,
                sample_weight,
            )
        else:
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[:-1], residY, residX[-1], beta_ls, sample_weight
            )
