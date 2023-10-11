import numpy as np
import time

class NewtonSolver:
    class solver_options:
        _tol: float = 1e-9
        _MaxIter_Newton: int = 100
        _MaxIter_LS: int = 20

    class SolverReport:
        _time_solver = 0.
        _times_called = 0
        _num_iterations = 0.

        # for the last call
        _ghist = []
        _fail_strike = 0

    def __init__(self, verbose = True):
        self.verbose = verbose
        self.solver_report = self.SolverReport() if verbose else None

    def solve(self, x: np.ndarray, func, func_with_derivatives):
        t_newton_start = time.time()

        _ndof_r = len(x)
        if _ndof_r == 0:
            return

        tol = self.solver_options._tol
        MaxIter_Newton = max(20 * _ndof_r, self.solver_options._MaxIter_Newton)
        MaxIter_LS = self.solver_options._MaxIter_LS

        MaxIter_LS_Fail_Strike = 10

        success_newton = False
        g_last = 1e5
        fail_strike = 0


        if self.solver_report is not None:
            g_his = []

        for iter_newton in range(MaxIter_Newton):
            g, H = func_with_derivatives(x)

            dx = np.linalg.solve(H, -g)

            g_new = func(x + dx)

            gnorm = np.linalg.norm(g)
            alpha = 1.0
            success_ls = False

            for trial in range(MaxIter_LS):
                g_new_trial = func(x + alpha * dx)

                if np.linalg.norm(g_new_trial) < gnorm:
                    success_ls = True
                    break

                alpha *= 0.5

            if success_ls:
                fail_strike = 0
            else:
                fail_strike += 1
                if fail_strike >= MaxIter_LS_Fail_Strike:
                    break

            x += alpha * dx

            if np.linalg.norm(g_new) < tol:
                success_newton = True
                break

            g_last = np.linalg.norm(g_new)
            if self.solver_report is not None:
                g_his.append(g_last)

        if not success_newton and (g_last > 1e-5 or np.isnan(g_last)):
            if self.verbose:
                print("Newton method did not converge. g =", g_last)
        if self.solver_report is not None:
            self.solver_report._time_solver += time.time() - t_newton_start
            self.solver_report._num_iterations += iter_newton
            self.solver_report._times_called += 1
            self.solver_report._ghist = g_his
            self.solver_report._fail_strike = fail_strike
        # print(f"{iter_newton} newton steps")
        return x

