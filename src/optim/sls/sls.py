import torch
import copy
import time

from . import utils as ut


class Sls(torch.optim.Optimizer):
    """Implements stochastic line search
    `paper <https://arxiv.org/abs/1905.09997>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        beta_f (float, optional): factor used by Goldstein for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        eta_max (float, optional): an upper bound used by Goldstein on the step size (default: 10)
        bound_step_size (bool, optional): a flag used by Goldstein for whether to bound the step-size (default: True)
        line_search_fn (float, optional): the condition used by the line-search to find the
                    step-size (default: Armijo)
    """

    def __init__(
        self,
        params,
        n_batches_per_epoch=500,
        c=1e-2,
        beta_b=0.8,
        gamma=5.0,
        beta_f=2.0,
        reset_option=1,
        eta_max=10,
        bound_step_size=True,
        line_search_fn="armijo",
        lr=1,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            n_batches_per_epoch=n_batches_per_epoch,
            c=c,
            beta_b=beta_b,
            gamma=gamma,
            beta_f=beta_f,
            reset_option=reset_option,
            eta_max=eta_max,
            bound_step_size=bound_step_size,
            line_search_fn=line_search_fn,
            lr=lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            step_size=lr,
        )
        super().__init__(params, defaults)
        self.params = dict()

        self.params["step"] = 0

        self.params["n_forwards"] = 0
        self.params["n_backwards"] = 0

        self._init_state()


    def _init_state(self):
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                state = self.state[p]

                state["step_t"] = (
                    torch.tensor(0.0, dtype=torch.float32)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )


    @torch.no_grad()
    def step(self, closure):
        # deterministic closure
        seed = time.time()

        @torch.no_grad()
        def closure_deterministic():
            self.params["n_forwards"] += 1
            with ut.random_seed_torch(int(seed)):
                return closure()

        # get loss
        loss = closure_deterministic()

        self.params["n_backwards"] += 1

        # loop over parameter groups
        for group in self.param_groups:
            params = group["params"]

            # save the current parameters:
            params_current = copy.deepcopy(params)
            grad_current = ut.get_adam_grad_list(group, self.state)

            grad_norm = ut.compute_grad_norm((p.grad for p in group["params"]))

            step_size = ut.reset_step(
                step_size=group["step_size"],
                n_batches_per_epoch=group["n_batches_per_epoch"],
                gamma=group["gamma"],
                reset_option=group["reset_option"],
                init_step_size=group["lr"],
            )

            # only do the check if the gradient norm is big enough
            if grad_norm >= 1e-8:
                # check if condition is satisfied
                found = 0
                step_size_old = step_size

                for e in range(10):
                    # try a prospective step
                    ut.try_sgd_update(
                        params, step_size, params_current, grad_current
                    )

                    if group["line_search_fn"] == "dummy":
                        found = 1
                        break

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()

                    # =================================================
                    # Line search
                    if group["line_search_fn"] == "armijo":
                        armijo_results = ut.check_armijo_conditions(
                            step_size=step_size,
                            step_size_old=step_size_old,
                            loss=loss,
                            grad_norm=grad_norm,
                            loss_next=loss_next,
                            c=group["c"],
                            beta_b=group["beta_b"],
                        )
                        found, step_size, step_size_old = armijo_results
                        if found == 1:
                            break

                    elif group["line_search_fn"] == "goldstein":
                        goldstein_results = ut.check_goldstein_conditions(
                            step_size=step_size,
                            loss=loss,
                            grad_norm=grad_norm,
                            loss_next=loss_next,
                            c=group["c"],
                            beta_b=group["beta_b"],
                            beta_f=group["beta_f"],
                            bound_step_size=group["bound_step_size"],
                            eta_max=group["eta_max"],
                        )

                        found = goldstein_results["found"]
                        step_size = goldstein_results["step_size"]

                        if found == 3:
                            break

                # if line search exceeds max_epochs
                if found == 0:
                    ut.try_sgd_update(params, 1e-6, params_current, grad_current)

            # save the new step-size
            group["step_size"] = step_size
        self.params["step"] += 1

        return loss
