import time
import warnings
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
from scipy.stats import qmc
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.util.termination.no_termination import NoTermination
from pymoo.interface import sample
from pymoo.factory import get_sampling


def init(pop: int,  relu_no: int, com_no: int, xl: int, xu: int) -> np.ndarray:
    """
    Initialize degrees of subfunctions (LHC sampling)
    :param pop: population size
    :param relu_no: num of relu
    :param com_no: num of composite sub-polynomials
    :param xl: lower bound
    :param xu: upper bound
    :return: random sampling
    """
    sampler = qmc.LatinHypercube(relu_no * com_no)
    sample = sampler.random(pop)
    sample = qmc.scale(sample, [xl] * relu_no * com_no, [xu] * relu_no * com_no).round().astype(int)
    ind = np.argwhere(np.logical_and(sample % 2 == 0, sample != 0))
    sample[ind[:, 0], ind[:, 1]] = sample[ind[:, 0], ind[:, 1]] - 1

    return sample


def select(fronts, NP, pressure=3):
    ranks = [np.ones_like(front) * i for i, front in enumerate(fronts)]
    fronts = np.concatenate(fronts)
    ranks = np.concatenate(ranks)
    selected = np.random.choice(len(fronts), (NP, pressure))
    fitness = ranks[selected]
    index = np.argmin(fitness, axis=1)
    offspring = [selected[i][j] for i, j in enumerate(index)]
    return offspring


class Mutation(object):
    def __init__(self, relu_no: int, xl: float, xu: float, decrease_prob: float = 0.5, increase_prob: float = 0.3):
        super(Mutation, self).__init__()
        self.relu_no = relu_no
        self.xl = xl
        self.xu = xu
        self.decrease_prob = decrease_prob
        self.increase_prob = increase_prob
        assert decrease_prob + increase_prob <= 1

    def __call__(self, X):

        Y = X.copy()
        N = Y.shape[0]
        relu_no = self.relu_no
        com_no = Y.shape[1] // relu_no
        Y[Y == 0] = -1
        do_mutate = np.random.randint(0, com_no, (N, relu_no))
        mask = np.zeros((N, relu_no * com_no), dtype=int)
        for i in range(N):
            for j in range(relu_no):
                mask[i, j * com_no + do_mutate[i, j]] = 1
        delta = np.random.random(Y.shape)
        delta[delta > 1 - self.increase_prob] = 2
        delta[delta <= self.decrease_prob] = -2
        delta = delta.astype(int)
        Y = Y + delta * mask
        Y[Y < self.xl] = self.xl
        Y[Y > self.xu] = self.xu

        return Y


class Crossover(object):
    def __init__(self, relu_no: int):
        super(Crossover, self).__init__()
        self.relu_no = relu_no

    def __call__(self, X):

        Y = X.copy()
        relu_no = self.relu_no
        Z = []
        for p1, p2 in zip(Y[0::2], Y[1::2]):
            p1 = np.array_split(p1, relu_no)
            p2 = np.array_split(p2, relu_no)
            o1 = []
            o2 = []
            for g1, g2 in zip(p1, p2):
                if np.random.random() < 0.5:
                    o1.append(g1)
                    o2.append(g2)
                else:
                    o1.append(g2)
                    o2.append(g1)
            o1 = np.concatenate(o1)
            o2 = np.concatenate(o2)
            Z.append(o1)
            Z.append(o2)

        Z = np.row_stack(Z)

        return Z


def get_tradeoffs(fit: np.ndarray, n: int = None):
    """
    :param fit: fitness, Mx2
    :param n: Top-n solutions
    :return: top-n fronts
    """
    ndom_sort = NonDominatedSorting()
    fronts = ndom_sort.do(fit)
    fronts_num = np.cumsum([f.size for f in fronts])
    if n is None or not any(fronts_num >= n):
        return fronts
    else:
        K = np.argwhere(fronts_num >= n)[0][0]
        fronts_selected = []
        for i in range(K + 1):
            if i == K and fronts_num[i] > n:
                dist = calc_crowding_distance(fit[fronts[i], :])
                end = n - fronts_num[i - 1] if i > 0 else n
                idx = np.argsort(dist)[::-1][0: end]
                fronts_selected.append(fronts[i][idx])
            else:
                fronts_selected.append(fronts[i])
    return fronts_selected


def rccde(degrees, d_max: int = 31, coeff_space: tuple = (-5, 5), scale_space: tuple = (1, 5),
          cycle: int = 100, decay: float = 1e-2, eps: float = 1e-6, steady_max: int = 20, verbose: bool = False):
    """
    Regularized Cooperative Coevolution Differential Evolution (R-CCDE)-based Composite Polynomial's coefficient solver
    :param degrees: degrees of composite polynomials
    :param d_max: maximum degree
    :param coeff_space: coefficient search space regarding of basis
    :param scale_space: scaling search space regarding of composite polynomial
    :param cycle: number of cycles to coevolution (default: 50)
    :param decay: scaling decay (default: 1e-2)
    :param eps: minimum numbers for coefficients (for numerical issue)
    :param steady_max: maximum number of steady cycles
    :param verbose: display progress
    :return: com_deg, com_coeff, err, degen
    """
    if type(degrees) == list:
        degrees = np.asarray(degrees, dtype=int)
    degrees = degrees[degrees > 0]
    deg = np.cumprod(degrees)[-1] if any(degrees) else 0
    domain = (-1, 1)
    if deg > 0:
        assert deg % 2 == 1, 'Degree of composite polynomials is not odd.'
        ctxt_poly = []
        ctxt_scale = []
        problem_coeff = []
        problem_scale = []
        algorithm_coeff = []
        algorithm_scale = []
        for i_, deg_ in enumerate(degrees):
            var_num_ = (deg_ + 1) // 2
            pop_size_ = 20 * var_num_
            problem_ = RCCDECoeff(ctxt_poly, ctxt_scale, i_, deg_, coeff_space)
            init_ = sample(get_sampling('real_lhs'), pop_size_, var_num_, xl=coeff_space[0], xu=coeff_space[1])
            init_ = init_ / abs(coeff_space[1])
            coef_ = init_[np.random.choice(pop_size_)]
            coef_ = np.row_stack((np.zeros(var_num_), coef_)).T.flatten()
            ctxt_poly.append(Chebyshev(coef_))
            algo_ = DE(pop_size=pop_size_, sampling=init_, variant="DE/rand/1/bin",
                       dither='vector', F=0.5, CR=0.5, jitter=True)
            algo_.setup(problem_, termination=NoTermination(), verbose=False)
            problem_coeff.append(problem_)
            algorithm_coeff.append(algo_)

            if i_ < degrees.size - 1:
                pop_size_ = 20
                problem_ = RCCDEScale(ctxt_poly, ctxt_scale, i_, scale_space, decay)
                init_ = sample(get_sampling('real_lhs'), pop_size_, 1, xl=scale_space[0], xu=scale_space[1])
                ctxt_scale.append(np.random.choice(init_.flatten()))
                algo_ = DE(pop_size=pop_size_, sampling=init_, variant="DE/rand/1/bin",
                           dither='vector', F=0.1, CR=0.5, jitter=True)
                algo_.setup(problem_, termination=NoTermination(), verbose=False)
                problem_scale.append(problem_)
                algorithm_scale.append(algo_)
            else:
                ctxt_scale.append(1.0)
                problem_scale.append(None)
                algorithm_scale.append(None)

        end = time.time()
        best_err = float('inf')
        steady_count = 0
        for c_ in range(cycle):
            input_ = np.random.uniform(low=domain[0], high=domain[1], size=100)
            input_ = np.append(input_, [domain[0], domain[1]])
            target_ = sign_slack_scaled(input_)
            for i_ in range(degrees.size):
                # evolve coefficient
                problem_ = problem_coeff[i_]
                algo_ = algorithm_coeff[i_]
                for j_ in range(problem_.n_var):
                    pop_ = algo_.ask()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        algo_.evaluator.eval(problem_, pop_, input=input_, target=target_)
                    algo_.tell(infills=pop_)
                res_ = algo_.result()
                if res_ is not None:
                    coef_ = res_.X
                    coef_ = np.row_stack((np.zeros(problem_.n_var), coef_)).T.flatten()
                    ctxt_poly[i_].coef = coef_  # update context

                # evolve scale
                if i_ < degrees.size - 1:
                    problem_ = problem_scale[i_]
                    algo_ = algorithm_scale[i_]
                    for j_ in range(problem_.n_var):
                        pop_ = algo_.ask()
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            algo_.evaluator.eval(problem_, pop_, input=input_, target=target_)
                        algo_.tell(infills=pop_)
                    res_ = algo_.result()
                    if res_ is not None:
                        ctxt_scale[i_] = res_.X.item()  # update context
            curr_err = res_.F.item()
            if curr_err < best_err:
                best_err = curr_err
                steady_count = 0
            else:
                steady_count += 1
                if steady_count > steady_max:
                    break
            if verbose:
                print('=> Cycle {:d}/{:d}, Time {:.2f}s, Current Error {:.6e}, Best Error {:.6e}'.
                      format(c_, cycle, time.time() - end, res_.F.item(), best_err))
        # convert to power basis, integrate coefficients and scales
        for i_ in range(degrees.size):
            poly_ = ctxt_poly[i_]
            poly_ = poly_.convert(kind=Polynomial)
            coef_ = poly_.coef / ctxt_scale[i_]
            poly_.coef = coef_
            ctxt_poly[i_] = poly_
        for i_ in range(len(ctxt_poly)):
            poly_ = ctxt_poly[i_]
            coef_ = poly_.coef
            if len(np.argwhere(np.abs(coef_) > eps)) > 0:
                deg_ = np.max(np.argwhere(np.abs(coef_) > eps))
                if deg_ < poly_.degree():
                    poly_ = poly_.cutdeg(deg_)
                    ctxt_poly[i_] = poly_
    else:
        ctxt_poly = [Polynomial([0.])]

    # merge
    com_poly = []
    for poly_ in ctxt_poly:
        if len(com_poly) == 0:
            com_poly.append(poly_)
        else:
            if poly_.degree() * com_poly[-1].degree() <= d_max:
                com_poly[-1] = poly_(com_poly[-1])
            else:
                com_poly.append(poly_)

    # evaluation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = np.linspace(domain[0], domain[1], 1000)
        y = np.maximum(x, 0)
        y_pred = (0.5 + eval_composite_polynomial(com_poly, x)) * x
        err = abs(y - y_pred).mean()
        if not np.isfinite(err):
            err = float('inf')

    return com_poly, err


class RCCDECoeff(Problem):
    def __init__(self, ctxt_poly: list, ctxt_scale: list, pos: int, deg: int, search_space: tuple):
        self.ctxt_poly = ctxt_poly
        self.ctxt_scale = ctxt_scale
        self.pos = pos
        self.n_var = (deg + 1) // 2
        super(RCCDECoeff, self).__init__(n_var=self.n_var, n_obj=1, xl=search_space[0], xu=search_space[1], type=float)

    def _evaluate(self, x, out, *args, **kwargs):
        input = kwargs.get('input')
        target = kwargs.get('target')
        x = np.stack((np.zeros(x.shape), x), axis=2)
        err = []
        ctxt_poly = self.ctxt_poly.copy()
        ctxt_scale = self.ctxt_scale.copy()
        for coef_ in x:
            coef_ = coef_.flatten()
            ctxt_poly[self.pos] = Chebyshev(coef_)
            pred = input.copy()
            try:
                for poly_, scal_ in zip(ctxt_poly, ctxt_scale):
                    pred = poly_(pred) / scal_
                err_ = abs(pred - target).mean()
                if not np.isfinite(err_):
                    err_ = float('inf')
            except OverflowError:
                err_ = float('inf')
            err.append(err_)
        out['F'] = np.asarray(err)


class RCCDEScale(Problem):
    def __init__(self, ctxt_poly: list, ctxt_scale: list, pos: int, search_space: tuple, decay: float):
        self.ctxt_poly = ctxt_poly
        self.ctxt_scale = ctxt_scale
        self.pos = pos
        self.decay = decay
        super(RCCDEScale, self).__init__(n_var=1, n_obj=1, xl=search_space[0], xu=search_space[1], type=float)

    def _evaluate(self, x, out, *args, **kwargs):
        input = kwargs.get('input')
        target = kwargs.get('target')
        err = []
        ctxt_poly = self.ctxt_poly.copy()
        ctxt_scale = self.ctxt_scale.copy()
        for s_ in x:
            ctxt_scale[self.pos] = s_
            pred = input.copy()
            try:
                for poly_, scal_ in zip(ctxt_poly, ctxt_scale):
                    pred = poly_(pred) / scal_
                err_ = abs(pred - target).mean()
                err_ += self.decay * s_ ** 2
                err_ = err_.item()
                if not np.isfinite(err_):
                    err_ = float('inf')
            except OverflowError:
                err_ = float('inf')
            err.append(err_)

        out['F'] = np.asarray(err)


def sign_slack_scaled(x: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    assert epsilon >= 0
    if epsilon == 0:
        return np.sign(x) * 0.5
    else:
        y = np.sign(x)
        ind = np.argwhere(np.logical_and(x >= -epsilon, x <= epsilon)).flatten()
        y[ind] = x[ind] / epsilon
        return y * 0.5


def eval_composite_polynomial(polys: list, input: np.ndarray) -> np.ndarray:
    pred = input.copy()
    for poly_ in polys:
        pred = poly_(pred)
    return pred
