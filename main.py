import time
import random

import torch
import scipy
import numpy as np

from torch import nn


def set_all_seeds(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True


def solve_double_ldl(
        g_abcd: torch.tensor, iv_mem: torch.tensor, i_src: list[torch.tensor],
        g_wl: torch.tensor, g_src_l: torch.tensor, g_src_r: torch.tensor,
        g_bl: torch.tensor, g_src_t: torch.tensor, g_src_b: torch.tensor,
        multiply_by_n=True
):
    """ Computes y = M.inv()@N@x = (L@D@L.T).inv()@N@x
    Solves two special tri-diagonals in M with LDL decomposition:
        M^-1 = ((L@D)@L.T).inv() = L.T.inv() @ (L@D).inv() """
    with torch.no_grad():
        (m, n), s, alpha = g_abcd.size(), iv_mem.size(-1), g_abcd.max()
        gw_src, gw_line, gw_out = g_src_l, g_wl, (g_src_r if torch.is_tensor(g_src_r) else 0.)
        gb_src, gb_line, gb_out = (g_src_t if torch.is_tensor(g_src_t) else 0.), g_bl, g_src_b
        tmp_m, tmp_n = iv_mem[-1].view(-1, s)[-m:], iv_mem[-1].view(-1, s)[-n - m:-m]
        d_n, d_m = iv_mem[-1].view(-1)[:n], iv_mem[-1].view(-1)[n:n + m]
        gw_diag, gb_diag = 2. + alpha/gw_line, 2. + alpha/gb_line

        # # Compute N@i_src --> Solve L@D for v_bl
        d_m[0] = 1. + (gb_src + alpha)/gb_line
        if multiply_by_n:
            tmp_n.copy_(iv_mem[0, 0]).sub_(iv_mem[1, 0]).mul_(g_abcd[0].view(n, 1))		# tmp = (v_wl - v_bl) @ -B
            iv_mem[0, 0].mul_(alpha).sub_(tmp_n)										# alpha*v_wl - tmp
            iv_mem[1, 0].mul_(alpha).add_(tmp_n)										# alpha*v_bl + tmp
        if torch.is_tensor(i_src[2]):
            iv_mem[1, 0].add_(i_src[2])
        iv_mem[1, 0].div_(d_m[0])
        for r in range(m - 2):
            d_m[r + 1] = gb_diag - d_m[r].reciprocal()
            if multiply_by_n:
                tmp_n.copy_(iv_mem[0, r + 1]).sub_(iv_mem[1, r + 1]).mul_(g_abcd[r + 1].view(n, 1))
                iv_mem[0, r + 1].mul_(alpha).sub_(tmp_n)
                iv_mem[1, r + 1].mul_(alpha).add_(tmp_n)
            iv_mem[1, r + 1].add_(iv_mem[1, r]).div_(d_m[r + 1])
        d_m[-1] = 1. + (gb_out + alpha)/gb_line - d_m[-2].reciprocal()
        if multiply_by_n:
            tmp_n.copy_(iv_mem[0, -1]).sub_(iv_mem[1, -1]).mul_(g_abcd[-1].view(n, 1))
            iv_mem[0, -1].mul_(alpha).sub_(tmp_n)
            iv_mem[1, -1].mul_(alpha).add_(tmp_n)
        if torch.is_tensor(i_src[3]):
            iv_mem[1, -1].add_(i_src[3])
        iv_mem[1, -1].add_(iv_mem[1, -2]).div_(d_m[-1])
        # Solve L.T for v_bl
        for r in range(2, m + 1):
            iv_mem[1, -r].add_(tmp_n.copy_(iv_mem[1, 1 - r]).div_(d_m[-r]))
        # i_mem[1, -r].add_(i_mem[0, 1 - r].div(d_n[-r]))
        iv_mem[1].div_(gb_line)

        # # Solve L@D for v_wl
        d_n[0] = 1. + (gw_src + alpha)/gw_line
        if torch.is_tensor(i_src[0]):
            iv_mem[0, :, 0].add_(i_src[0])
        iv_mem[0, :, 0].div_(d_n[0])
        for r in range(n - 2):
            d_n[r + 1] = gw_diag - d_n[r].reciprocal()
            iv_mem[0, :, r + 1].add_(iv_mem[0, :, r]).div_(d_n[r + 1])
        d_n[-1] = 1. + (gw_out + alpha)/gw_line - d_n[-2].reciprocal()
        if torch.is_tensor(i_src[1]):
            iv_mem[0, :, -1].add_(i_src[1])
        iv_mem[0, :, -1].add_(iv_mem[0, :, -2]).div_(d_n[-1])
        # Solve L.T for v_wl
        for r in range(2, n + 1):
            iv_mem[0, :, -r].add_(tmp_m.copy_(iv_mem[0, :, 1 - r]).div_(d_n[-r]))
        iv_mem[0].div_(gw_line)
    return iv_mem


def frobenius_diff(
        g_abcd: torch.tensor, iv_mem: torch.tensor, i_src: list[torch.tensor],
        g_wl: torch.tensor, g_src_l: torch.tensor, g_src_r: torch.tensor,
        g_bl: torch.tensor, g_src_t: torch.tensor, g_src_b: torch.tensor
):
    """ Computes Frobenius norm of the difference between the input data and the matrix
        multiplication of G_ABCD and the outputs """
    with torch.no_grad():
        _, m, n, s = iv_mem.size()
        g = g_abcd.view(m, n, 1)  # G = [[A, B], [C, D]]
        v_wl, v_bl = iv_mem[:2]  # V = [[v_wl], [v_bl]]
        tmp = iv_mem[-1]

        # F = [[A, B], [0, 0]]@V + ...
        tmp[:, :] = -g*v_bl[:, :]			# B.diag()@v_bl
        tmp[:, 1:] -= g_wl*v_wl[:, :-1]		# A.tril(-1)@v_wl
        tmp[:, :-1] -= g_wl*v_wl[:, 1:]		# A.triu(1)@v_wl
        # A.diag()*v_wl
        tmp[:, 0] += (g[:, 0] + g_src_l + g_wl)*v_wl[:, 0]
        tmp[:, 1:-1] += (g[:, 1:-1] + 2*g_wl)*v_wl[:, 1:-1]
        tmp[:, -1] += (g[:, -1] + g_src_r + g_wl)*v_wl[:, -1]
        if i_src[0] is not None:
            tmp[:, 0].sub_(i_src[0])
        if i_src[1] is not None:
            tmp[:, n - 1].sub_(i_src[1])
        norm = tmp.pow_(2).sum()  # .sum(dim=(0, 1))  # .detach()

        # F = ... + [[0, 0], [C, D]]@V
        tmp[:] = -g*v_wl[:]				# C.diag()@v_wl
        tmp[1:] -= g_bl*v_bl[:-1]		# D.tril(-1)@v_bl
        tmp[:-1] -= g_bl*v_bl[1:]		# D.triu(1)@v_bl
        # D.diag()*v_bl
        tmp[0] += (g[0] + g_src_t + g_bl)*v_bl[0]
        tmp[1:-1] += (g[1:-1] + 2*g_bl)*v_bl[1:-1]
        tmp[-1] += (g[-1] + g_src_b + g_bl)*v_bl[-1]
        if i_src[2] is not None:
            tmp[0].sub_(i_src[2])
        if i_src[3] is not None:
            tmp[-1].sub_(i_src[3])
        norm += tmp.pow_(2).sum()  # .sum(dim=(0, 1))
    return norm.sqrt().item()  # .sum().sqrt().item()  # ||G@V - i_src||_F


def solve_xbar(
        g_abcd: torch.tensor, iv_mem: torch.tensor, i_src: list[torch.tensor],
        g_wl: torch.tensor, g_src_l: torch.tensor, g_src_r: torch.tensor,
        g_bl: torch.tensor, g_src_t: torch.tensor, g_src_b: torch.tensor,
        r_tol: float = None, max_it: int = None, max_time: int = None,
        slow: float = 2**-30, log: bool = False,
):
    def r(g):
        return g if torch.is_tensor(g) else 0.  # TODO - torch.tensor([g]) or fail

    max_time = max_time or int(100e9)
    if max_time is None and max_it is None and r_tol is None:
        r_tol = 1e-6
    t, it, r_err = time.perf_counter_ns(), 0, 1.
    (m, n), s, alpha = g_abcd.size(), iv_mem.size(-1), g_abcd.max()
    g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b = r(g_wl), r(g_src_l), r(g_src_r), r(g_bl), r(g_src_t), r(g_src_b)

    norm = torch.zeros(1, device=i_src[0].device)
    for i in i_src:
        if torch.is_tensor(i):
            norm += i.norm(p='fro')**2
    norm = norm.sqrt().item()
    err_l = [(norm, -1), (r_err, 0)]
    solve_double_ldl(g_abcd, iv_mem[:3].zero_(), i_src, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b, False)
    while time.perf_counter_ns() - t < max_time:  # for k in range(max(20, 3*m + n)):
        # -- v[k + 1] = M.inv()@(N@v[k] + i_src) --
        solve_double_ldl(g_abcd, iv_mem, i_src, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b, multiply_by_n=True)
        if (torch.is_tensor(i_src[0]) and i_src[0].isnan().any()) or \
                (torch.is_tensor(i_src[1]) and i_src[1].isnan().any()) or \
                (torch.is_tensor(i_src[2]) and i_src[2].isnan().any()) or \
                (torch.is_tensor(i_src[3]) and i_src[3].isnan().any()) or \
                iv_mem[:2].isnan().any() or g_abcd.isnan().any():
            print('NaN', i_src, iv_mem[:2], g_abcd)
        it += 1
        r_err = frobenius_diff(g_abcd, iv_mem, i_src, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b)/norm
        err_l.append((r_err, time.perf_counter_ns() - t))
        if (r_tol and r_err < r_tol) or (max_it and it >= max_it) or (test := err_l[-2][0]/r_err - 1 < slow):
            # print(test)
            break
    # print(f'iter: {k}, F_norm: {r_err:.3e}, time: {(time.perf_counter_ns() - tim)*1e-9:.5f}s ')
    return iv_mem, err_l if log else r_err, it


class XBarSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, g_abcd: torch.tensor, i_mem: torch.tensor, i_src: torch.tensor,
        g_wl: torch.tensor, g_src_l: torch.tensor, g_src_r: torch.tensor,
        g_bl: torch.tensor, g_src_t: torch.tensor, g_src_b: torch.tensor,
        err_threshold: float = 1e-4, *args, **kwargs
    ):
        with torch.no_grad():
            _, m, n, s = i_mem.size()
            in_src = [None, None, None, None]
            dev, dtp = g_abcd.device, g_abcd.dtype

            # Word Lines
            if i_src.size(0) == m:
                in_src[0] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src).mul_(g_src_l)
            elif i_src.size(0) == 2*m and (torch.is_tensor(g_src_r) or m != n):
                in_src[0] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[0::2]).mul_(g_src_l)
                in_src[1] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[1::2]).mul_(g_src_r)
            # Bit Lines (Backward)
            elif i_src.size(0) == n:
                in_src[3] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src).mul_(g_src_b)
            elif i_src.size(0) == 2*n:
                in_src[2] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[:n]).mul_(g_src_t)
                in_src[3] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[n:]).mul_(g_src_b)
            # Word + Bit Lines
            elif i_src.size(0) == (m + n):
                in_src[0] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[:m]).mul_(g_src_l)
                in_src[3] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[-n:]).mul_(g_src_b)
            elif i_src.size(0) == 2*m + n:
                in_src[0] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[0:2*m:2]).mul_(g_src_l)
                in_src[1] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[1:2*m:2]).mul_(g_src_r)
                in_src[3] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[-n:]).mul_(g_src_b)
            elif i_src.size(0) == 2*(m + n):
                in_src[0] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[0:2*m:2]).mul_(g_src_l)
                in_src[1] = torch.empty((m, s), device=dev, dtype=dtp).copy_(i_src[1:2*m:2]).mul_(g_src_r)
                in_src[2] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[-2*n:-n]).mul_(g_src_t)
                in_src[3] = torch.empty((n, s), device=dev, dtype=dtp).copy_(i_src[-n:]).mul_(g_src_b)
            # Raw input
            elif i_src.size(0) == 2*m*n:
                in_src[0], in_src[1], in_src[2], in_src[3] = [None, None, None, None]
                raise NotImplementedError
            else:
                raise IndexError

            v_out, *_ = solve_xbar(g_abcd, i_mem, in_src, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b)
            ctx.save_for_backward(g_abcd, v_out, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b)
            ctx.grad_i_size, ctx.threshold = [(i is not None) for i in in_src], err_threshold
        return v_out[1, -1].clone().requires_grad_(True)

    @staticmethod
    def backward(ctx, *grad):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        with torch.no_grad():
            grad_v_bl, *_ = grad
            err_threshold, grad_i_size = ctx.threshold, ctx.grad_i_size
            g_abcd, v_out, g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b = ctx.saved_tensors

            _, m, n, s = v_out.size()
            grad_i, *_ = solve_xbar(g_abcd, v_out[2:], [None, None, None, grad_v_bl],
                                g_wl, g_src_l, g_src_r, g_bl, g_src_t, g_src_b)

            # (-grad_i@v_wb.T).diag() + ().diag(m*n) + ().diag(-m*n)
            grad_g = torch.einsum('bijk,bijk->ij', grad_i[:2], v_out[:2])  # grad_A + grad_D
            grad_g += torch.einsum('ijk,ijk->ij', grad_i[1], v_out[0])  # + grad_B
            grad_g += torch.einsum('ijk,ijk->ij', grad_i[0], v_out[1])  # + grad_C

            grad_i_ret = torch.zeros(
                sum(a*b for a, b in zip((m, m, n, n), grad_i_size)), s, device=grad_i.device, dtype=grad_i.dtype)
            if grad_i_size[0]:
                if grad_i_size[1]:
                    grad_i_ret[0:2*m:2].copy_(grad_i[0, :, 0])
                    grad_i_ret[1:2*m:2].copy_(grad_i[0, :, -1])
                else:
                    grad_i_ret[:m].copy_(grad_i[0, :, 0])
            else:
                if grad_i_size[1]:
                    grad_i_ret[:m].copy_(grad_i[0, :, -1])
            if grad_i_size[3]:
                if grad_i_size[2]:
                    grad_i_ret[-2*n:-n].copy_(grad_i[0, 0])
                grad_i_ret[-n:].copy_(grad_i[0, -1])
            else:
                if grad_i_size[2]:
                    grad_i_ret[-n:].copy_(grad_i[0, 0])
        return grad_g, None, grad_i_ret, None, None, None, None, None, None, None


class XbarLinear(nn.Linear):
    def __init__(
            self, *args, weight: torch.tensor = None, line_g: float = 1/2., src_g: float = 1/50.,
            min_g=1e-5, max_g=1e-3, lr_input=False, tb_output=False, **kwargs
    ):
        if weight is None and len(args) < 2:
            raise TypeError(
                "XbarLinear.__init__() missing at least 1 required argument: '__init__(in_features, out_features, ...)'"
                "or '__init__(weights=...)'\nYou can provide the size of layer [in, out], or the desired weight tensor."
            )
        with torch.no_grad():
            super(XbarLinear, self).__init__( *(args or list(weight.size()[::-1])) )  # , **kwargs)
            # Set weights
            self.min_g, self.max_g = min_g, max_g
            (m, n), dev, dtp = weight.size(), weight.device, weight.dtype
            if weight is not None:
                self.to(device=dev, dtype=dtp)
                self.weight.copy_(weight).contiguous()
            self.max_0, self.min_0 = self.weight.max(), self.weight.min()
            self.weight.add_(self.max_0).mul_((max_g - min_g)/(self.max_0 - self.min_0)).add_(min_g)

            # Set parasite resistances - Word/Bit Lines
            # (line_g if torch.is_tensor(line_g) else torch.tensor(line_g))
            # .requires_grad_(False).to(device=dev, dtype=dtp).view(1, 1)  # TODO - not 1x1
            self.g_wl = torch.tensor(line_g, requires_grad=False, device=dev, dtype=dtp).view(1, 1)
            self.wide_mode = (self.g_wl.size(1) == n + 1)
            if 'g_bl' in kwargs:
                self.g_bl = kwargs['g_bl'].to(device=dev, dtype=dtp)
                self.tall_mode = (self.g_bl.size(0) == m + 1)
            else:
                self.g_bl, self.tall_mode = self.g_wl, False

            # Set parasite resistances - Inputs/Outputs
            self.g_src_l = torch.tensor(src_g, requires_grad=False, device=dev, dtype=dtp).view(1, 1)
            self.g_src_r = self.g_src_l if lr_input else kwargs.get('g_src_r', None)
            if torch.is_tensor(self.g_src_r):
                self.g_src_r = self.g_src_r.to(device=dev, dtype=dtp)
            self.g_src_b = kwargs.get('g_src_b', self.g_src_l.mean()).to(device=dev, dtype=dtp).view(1, 1)
            self.g_src_t = self.g_src_b if tb_output else kwargs.get('g_src_t', None)
            if torch.is_tensor(self.g_src_t):
                self.g_src_t = self.g_src_t.to(device=dev, dtype=dtp)

            # Temporary memory
            self.iv_mem = None

    def forward(self, b: torch.Tensor, test=True) -> torch.Tensor:
        if self.iv_mem is None or self.iv_mem.size(-1) != b.size(-1):
            self.iv_mem = torch.empty(
                (5, *self.weight.size(), b.size(-1)), device=self.weight.device, dtype=self.weight.dtype)
        # with torch.no_grad():
        # 	self.weight.clamp_(1e-5, 1e-3)  # , 1e-3
        return XBarSolve.apply(
            self.weight.clamp(self.min_g), self.iv_mem, b, self.g_wl, self.g_src_l, self.g_src_r,  # , self.max_g
            self.g_bl, self.g_src_t, self.g_src_b).T  # + (0. if self.bias is None else self.bias)


TESTING = True


def tst_time_norm():
    from scipy.sparse.linalg import gmres, lgmres, LinearOperator
    from scipy.sparse import diags, kron, bmat, eye
    from scipy.linalg import solve_toeplitz
    torch.set_default_dtype(torch.float64)
    dev = torch.device('cpu')

    m_tim, size_min, size_max, num_tests, batch_size = \
        int(10e9), 7, 10, 1 if TESTING else 5, 1 if TESTING else 10
    weights = torch.randn(num_tests, 2**size_max, 2**size_max, device=dev)
    weights = weights - weights.min()
    weights = weights/weights.max()

    base_g_src_l, base_g_src_r, base_g_src_t, base_g_src_b, base_g_w, base_g_b = \
        torch.tensor([1, 1e-6, 1e-6, 1, 1, 1], device=dev)
    x = torch.randn(2**size_max, batch_size, device=dev)

    class acc:
        data, time, t = [], [], 0

    def identity_m(x_in):
        return x_in

    scipy_opt = [
        (gmres, {'tol': 1e-13, 'callback_type': 'x'}),
        (lgmres, {'atol': 1e-13})
    ]

    f = open(f'results/res_{time.time_ns()}.csv', 'w')
    tmp_mem = torch.zeros(3, 2**size_max, 2**size_max, batch_size, device=dev)
    f.write(f'method,\ttype,\tweights,\tscale,\tsize,\tpcond,'
            f'\tinput,\titer,\ttime,\tnorm,\terror\n')
    for i, w in enumerate(weights):
        for s in [7, 5, 3, 1, -1, -3, -5, -7, -9, -11][-3:]:
            scale = 2**s
            for m in list(range(size_min, size_max)):
                m = n = 2**m
                g = w[:m, :n]*scale

                g_w, g_src_l, g_src_r = base_g_w.item(), base_g_src_l.item(), base_g_src_r.item()
                g_b, g_src_t, g_out_b = base_g_b.item(), base_g_src_t.item(), base_g_src_b.item()
                gg_t, gg_b = diags(
                    [[-g_w]*(n - 1), [g_w + g_src_l, *[2*g_w]*(n - 2), g_w + g_src_r], [-g_w]*(n - 1)], [-1, 0, 1]
                ), diags(
                    [[-g_b]*(n - 1), [g_b + g_src_t, *[2*g_b]*(n - 2), g_b + g_out_b], [-g_b]*(n - 1)], [-1, 0, 1]
                )
                gg = kron(diags([[-1], [1, 1], [-1]], [-1, 0, 1]), diags([g.flatten()], [0]))
                tb = bmat([[kron(eye(m), gg_t), None], [None, kron(gg_b, eye(m))]])

                # Full size
                xx_sp, ABCD_sp = torch.zeros(2*m*n, device=dev), (tb + gg).tocsc()
                print(f'\n{scipy.sparse.linalg.norm(ABCD_sp, "fro")},\tfull,\t{i},\t{s},\t{m}x{n}\n')
                if TESTING:
                    continue

                J_inv = diags([1/ABCD_sp.diagonal(0)], [0])

                def jacobi_m(x_in):
                    return J_inv@x_in

                def calls(data):
                    acc.time.append(time.perf_counter_ns() - acc.t)
                    acc.data.append(torch.as_tensor(xx_sp - ABCD_sp@data, device=dev).norm('fro'))
                    if acc.time[-1] > m_tim:
                        raise TimeoutError

                for func, kwargs in scipy_opt:
                    for j, pcond in enumerate((identity_m, jacobi_m, )):  # inverse_m,
                        pcond = LinearOperator((2*m*n, 2*m*n), pcond)
                        print(f'\n{func.__name__},\tfull,\t{i},\t{s},\t{m}x{n},\t{j},')
                        for k, xx in enumerate(x.T):  # , start=batch_size - 1):
                            acc.data, acc.time = [], []
                            xx_sp[:m*n:n] = xx[:m]/xx[:m].norm()
                            acc.t = time.perf_counter_ns()
                            try:
                                y, err = func(ABCD_sp, xx_sp, callback=calls, M=pcond, **kwargs)
                                print('^', end='')
                            except TimeoutError:
                                err = len(acc.data)
                                print('-', end='')
                            for en, (dif, t) in enumerate(zip(acc.data, acc.time)):
                                f.write(f'{func.__name__},\tfull,\t{i},\t{s},\t{m}x{n},'
                                        f'\t{j},\t{k},\t{en + 1},\t{t},\t{dif:.6e},\t{err}\n')
                # Shur
                xx_sp = xx_sp[:m*n]
                g_mean, b_inv = g.mean(), diags([-1/g.flatten()], [0])
                ABCD_sp = -diags([g.flatten()], [0]) - kron(gg_b, eye(m))@b_inv@kron(eye(n), gg_t)
                J_inv = diags([1/ABCD_sp.diagonal(0)], [0])

                def jacobi_m(x_in):
                    return J_inv@x_in

                def solve_m(x_in):
                    x_sl = x_in[:m*n].reshape(m, n)
                    x_g2 = solve_toeplitz([2/g_mean + 1, -1] + [0]*(n - 2), x_sl).T
                    return solve_toeplitz([2 + g_mean, -1] + [0]*(m - 2), x_g2).reshape(*x_in.shape)

                def calls(data):
                    acc.time.append(time.perf_counter_ns() - acc.t)
                    acc.data.append(torch.as_tensor(xx_sp - ABCD_sp@data, device=dev).norm('fro'))
                    if acc.time[-1] > m_tim:
                        raise TimeoutError

                for func, kwargs in scipy_opt:
                    for j, pcond in enumerate((identity_m, jacobi_m, solve_m, )):
                        pcond = LinearOperator((m*n, m*n), pcond)
                        print(f'\n{func.__name__},\tshur,\t{i},\t{s},\t{m}x{n},\t{j},')
                        for k, xx in enumerate(x.T):
                            acc.data, acc.time = [], []
                            xx_sp[:m*n:n] = xx[:m]/xx[:m].norm()
                            acc.t = time.perf_counter_ns()
                            try:
                                y, err = func(ABCD_sp, xx_sp, callback=calls, M=pcond, **kwargs)
                                print('^', end='')
                            except Exception as exc:
                                print('-' if isinstance(exc, TimeoutError) else 'x', end='')
                                err = len(acc.data)
                            for it, (dif, t) in enumerate(zip(acc.data, acc.time)):
                                f.write(f'{func.__name__},\tshur,\t{i},\t{s},\t{m}x{n},'
                                        f'\t{j},\t{k},\t{it + 1},\t{t},\t{dif:.6e},\t{err}\n')  # """
                # Ours
                mem = tmp_mem.view(-1)[:3*m*n].view(3, m, n, 1).zero_()
                for r in range(batch_size - 1 if TESTING else 0, batch_size):
                    xx = x[:m, r:r + 1]/x[:m, r:r + 1].norm(dim=0)
                    y, diff, it = solve_xbar(g, mem, i_src=[xx, None, None, None],
                                             g_wl=base_g_w, g_src_l=base_g_src_l, g_src_r=base_g_src_r,
                                             g_bl=base_g_b, g_src_t=base_g_src_t, g_src_b=base_g_src_b,
                                             max_time=m_tim, log=True)
                    for it, (dif, t) in enumerate(diff[2:]):
                        f.write(f'ours,\tfull,\t{i},\t{s},\t{m}x{n},'
                                f'\t0,\t{r},\t{it + 1},\t{t},\t{dif:.6e},\t{1*(t > m_tim)}\n')

                if multi_vector := False:
                    mem = tmp_mem.view(-1)[:3*m*n*batch_size].view(3, m, n, batch_size).zero_()
                    xx = x[:m]/x[:m].norm(dim=0)
                    t = time.perf_counter_ns()
                    y, diff, it = solve_xbar(g, mem, i_src=[xx, None, None, None],
                                             g_wl=base_g_w, g_src_l=base_g_src_l, g_src_r=base_g_src_r,
                                             g_bl=base_g_b, g_src_t=base_g_src_t, g_src_b=base_g_src_b,
                                             max_time=m_tim, log=True)
                    t = time.perf_counter_ns() - t
                    print(f'\nOurs ({it} iter) [\t{i},\t{s},\t{(m, n)}]\n'
                          f'\tt = \t\t{t*1e-9:.6e} s\n\t2-norm =\t{diff[-1][0]:.6e}')
                    for it, (dif, t) in enumerate(diff[2:]):
                        f.write(f'ours,\tfull,\t{i},\t{s},\t{m}x{n},'
                                f'\t0,\t{batch_size},\t{it + 1},\t{t},\t{dif:.6e},\t{1*(t > m_tim)}\n')  # """
                f.flush()
            pass
        pass
    f.close()


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    set_all_seeds(621)
    tst_time_norm()
