"""Levenberg-Marquardt optimizer for joint DH + hand-eye calibration.

Basic LM loop with:
  - Row-wise weighting (position vs orientation)
  - L2 regularization toward initial parameters
  - SVD-based pseudo-inverse with optional observability masking
"""

import numpy as np


def levenberg_marquardt(residual_fn, xi0, w_rows, *,
                        max_iter=200, tol=1e-6, lam0=1e-3,
                        reg_weight=0.0, svd_thresh=0.0, verbose=True):
    """Generic LM solver.

    Parameters
    ----------
    residual_fn : callable(xk) -> (r_flat, J)
        Returns unweighted residual vector and Jacobian at xk.
    xi0 : ndarray
        Initial parameter vector (also the regularization anchor).
    w_rows : ndarray
        Per-row weights applied to residuals before cost computation.
    max_iter : int
    tol : float
        Convergence threshold on ||dx|| (default 1e-6, sufficient for
        mm-scale calibration — avoids grinding through noise floor).
    lam0 : float
        Initial Levenberg-Marquardt damping.
    reg_weight : float
        L2 penalty strength.  Zero disables regularization.
    svd_thresh : float
        Fraction of largest singular value; components below are zeroed.
        Zero disables masking.
    verbose : bool

    Returns
    -------
    dict with keys: xi, converged, iters, hist_pos, hist_ori, hist_cost, hist_lam
    """
    x = xi0.copy()
    lam = lam0
    n_params = len(x)
    l = len(w_rows) // 6

    # Per-pose error reporting: mean of norms (axis-angle for orientation)
    def _errors(rw):
        pos_norms = [np.linalg.norm(rw[i * 6:i * 6 + 3]) / w_rows[0] * 1000 for i in range(l)]
        ori_norms = [np.linalg.norm(rw[i * 6 + 3:i * 6 + 6]) / w_rows[3] * 180 / np.pi for i in range(l)]
        return np.mean(pos_norms), np.mean(ori_norms)

    def total_cost(xk):
        r, _ = residual_fn(xk)
        rw = r * w_rows
        data_cost = 0.5 * np.dot(rw, rw)
        reg_cost = 0.5 * reg_weight * np.dot(xk - xi0, xk - xi0) if reg_weight > 0 else 0.0
        return data_cost + reg_cost

    r_old, _ = residual_fn(x)
    rw_old = r_old * w_rows
    c_old = total_cost(x)

    if verbose:
        pe, oe = _errors(rw_old)
        print("=" * 65)
        print("  LM Joint Calibration — DH + Hand-Eye")
        print("=" * 65)
        print("  poses=%d  params=%d  equations=%d" % (l, n_params, l * 6))
        print("  weights: pos=%.0f  ori=%.0f  reg=%.0e  svd=%.0e" %
              (w_rows[0], w_rows[3], reg_weight, svd_thresh))
        print("  initial spread  pos=%.3f mm  ori=%.3f deg" % (pe, oe))
        print("-" * 65)
        print("%4s  %12s  %12s  %10s  %10s" % ("it", "lambda", "||dx||", "pos(mm)", "ori(deg)"))
        print("-" * 65)

    # --- Observability mask from initial Jacobian ---
    _, J0 = residual_fn(x)
    JW0 = J0 * w_rows[:, np.newaxis]
    obs_mask = np.ones(n_params, dtype=bool)
    if svd_thresh > 0:
        _, s0, _ = np.linalg.svd(JW0, full_matrices=False)
        obs_mask = s0 > svd_thresh * s0[0]
        n_obs = int(obs_mask.sum())
        if verbose and n_obs < n_params:
            print("  observable: %d/%d (freezing %d)" % (n_obs, n_params, n_params - n_obs))

    converged = False
    stalled = False
    prev_cost = total_cost(x)
    prev_dx_norm = None
    slow_count = 0
    hist_pe, hist_oe, hist_cost, hist_lam = [], [], [], []

    for it in range(1, max_iter + 1):
        _, J = residual_fn(x)
        Jw = J * w_rows[:, np.newaxis]

        # Build augmented system [Jw; sqrt(reg)*I] @ dx = [-rw; sqrt(reg)*(xi0-x)]
        if reg_weight > 0:
            reg_block = np.sqrt(reg_weight) * np.eye(n_params)
            Ja = np.vstack([Jw, reg_block])
            ra = np.concatenate([rw_old, np.sqrt(reg_weight) * (xi0 - x)])
        else:
            Ja = Jw
            ra = rw_old

        U, s, Vt = np.linalg.svd(Ja, full_matrices=False)
        d = s / (s ** 2 + lam)
        if svd_thresh > 0:
            d[~obs_mask] = 0.0
        dx = Vt.T @ (d * (U.T @ ra))

        c_prev = c_old
        xt = x + dx
        cn = total_cost(xt)

        if cn < c_old:
            x = xt
            r_new, _ = residual_fn(x)
            rw_old = r_new * w_rows
            c_old = cn
            # Adaptive lambda: bigger drop → reduce damping more aggressively
            improvement = (c_prev - cn) / (abs(c_prev) + 1e-15)
            if improvement > 0.1:
                lam_factor = 0.1   # great step, trust Gauss-Newton
            elif improvement > 0.01:
                lam_factor = 0.3
            else:
                lam_factor = 0.5   # marginal, be cautious
            lam = max(lam * lam_factor, 1e-12)
        else:
            lam *= 3.0
            if lam > 1e14:
                if verbose:
                    print("\n  lambda saturated at %.0e" % lam)
                break

        pe, oe = _errors(rw_old)
        hist_pe.append(pe)
        hist_oe.append(oe)
        hist_cost.append(c_old)
        hist_lam.append(lam)

        if verbose:
            print("%4d  %12.2e  %12.2e  %10.4f  %10.4f" %
                  (it, lam, np.linalg.norm(dx), pe, oe))

        dx_norm = np.linalg.norm(dx)

        # Convergence: step size small
        if dx_norm < tol:
            converged = True
            if verbose:
                print("\n  Converged (||dx|| < tol) at iteration %d" % it)
            break

        # Cost stalled
        rel_drop = (prev_cost - c_old) / (abs(prev_cost) + 1e-15)
        if rel_drop < 1e-10:
            stalled = True
            if verbose:
                print("\n  Stalled (cost drop < 1e-10) at iteration %d" % it)
            break
        prev_cost = c_old

        # Slow crawl detection: lambda near floor and ||dx|| barely changing
        if lam <= 1e-8 and prev_dx_norm is not None:
            dx_change = abs(dx_norm - prev_dx_norm) / (prev_dx_norm + 1e-15)
            if dx_change < 0.05:
                slow_count += 1
            else:
                slow_count = 0
            if slow_count >= 3:
                converged = True
                if verbose:
                    print("\n  Converged (slow crawl detected) at iteration %d" % it)
                break
        prev_dx_norm = dx_norm

    if not converged and not stalled and verbose:
        print("\n  Max iterations reached (%d)" % max_iter)
    if stalled and verbose:
        pass  # already printed above

    return {
        "xi": x,
        "converged": converged,
        "iters": it,
        "hist_pos": np.array(hist_pe),
        "hist_ori": np.array(hist_oe),
        "hist_cost": np.array(hist_cost),
        "hist_lam": np.array(hist_lam),
    }
