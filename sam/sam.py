import torch
from torch.optim.optimizer import Optimizer

class SAM(Optimizer):
    
    def __init__(self, params, base_optimizer: Optimizer, rho: float = 0.05):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}. Must be >= 0.")
        defaults = dict(rho=rho)
        
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm() 

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                #Compute epsilon_hat from parameters
                e_hat = self._compute_epsilon_hat(p.grad, grad_norm)

                # Store epsilon_hat in the state so second_step can undo the perturbation
                self.state[p]["e_hat"] = e_hat

                # Perturb the weights
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

                # Undo the perturbation: w - epsilon_hat -> w
                p.data.sub_(e_hat)

                # Delete stored perturbation
                del self.state[p]["e_hat"]

        # Let the base optimizer do its normal step
        self.base_optimizer.step() 

        if zero_grad:
            self.base_optimizer.zero_grad()
