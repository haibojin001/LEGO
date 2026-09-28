# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg204::torch.autograd.grad+torch.ones_like
# name: torch_primitive
# summary: Uses torch.autograd.grad, torch.ones_like across 2 repos
# anchor_symbols: ['torch.autograd.grad', 'torch.ones_like']
# observed in 2 repos: ['microsoft__RD-Agent', 'pykale__pykale']...

# --- from pykale__pykale::kale/evaluate/metrics.py::gradient_penalty ---
def gradient_penalty(critic, h_s, h_t):
    """Computes gradient penalty in Wasserstein distance guided representation learning

    Examples:
        See WDGRLTrainer and WDGRLTrainerMod in kale.pipeline.domain_adapter
    """

    alpha = torch.rand(h_s.size(0), 1)
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    # try:
    differences = h_t - h_s

    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(
        preds,
        interpolates,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
        create_graph=True,
    )[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    # except:
    #     gradient_penalty = 0

    return gradient_penalty

# --- from microsoft__RD-Agent::rdagent/components/coder/model_coder/benchmark/gt_code/visnet.py::ViSNet.forward ---
def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """
        if self.derivative:
            pos.requires_grad_(True)

        x, v = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

        if self.derivative:
            grad_outputs = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return y, -dy

        return y, None
