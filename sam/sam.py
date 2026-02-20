import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    def __init__(self, params, base_optimizer: Optimizer, rho: float = 0.05, p: int = 2):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}. Must be >= 0.")
        if p not in (2, float("inf")):
            raise ValueError(f"Unsupported p-norm: {p}. Use 2 or float('inf').")

        defaults = dict(rho=rho, p=p)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.p = p

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):

        grad_norm = self._grad_norm()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                e_hat = self._compute_epsilon_hat(p.grad, grad_norm)

                self.state[p]["e_hat"] = e_hat

                p.data.add_(e_hat)

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                e_hat = self.state[p].get("e_hat")
                if e_hat is None:
                    raise RuntimeError(
                        "second_step() called without a preceding first_step(). "
                        "Make sure first_step() was called after the first backward()."
                    )

                p.data.sub_(e_hat)

                del self.state[p]["e_hat"]

        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

  
    def _compute_epsilon_hat(self, grad: torch.Tensor, grad_norm: torch.Tensor) -> torch.Tensor:
        if self.p == 2:
            # Dual norm q = 2.  ε_hat = ρ * grad / ||grad||_2
            return self.rho * grad / (grad_norm + 1e-12)

        elif self.p == float("inf"):
            # Dual norm q = 1.  ε_hat = ρ * sign(grad)
            return self.rho * grad.sign()

    def _grad_norm(self) -> torch.Tensor:
        # Determine which dual norm q to compute (1/p + 1/q = 1)
        if self.p == 2:
            q = 2
        elif self.p == float("inf"):
            q = 1
        else:
            raise ValueError(f"Unsupported p: {self.p}")

        norm = torch.tensor(0.0, device=next(self.param_groups[0]["params"][0].device
                                              for _ in [None]))

        # Accumulate ||grad||_q^q across all parameters, then take the q-th root
        accum = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if accum is None:
                    accum = p.grad.norm(q).pow(q)
                else:
                    accum = accum + p.grad.norm(q).pow(q)

        if accum is None:
            return torch.tensor(0.0)

        return accum.pow(1.0 / q)
