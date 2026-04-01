import torch

class LCA_GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}

        # Handle DataParallel
        self.target_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        self.target_model.SA_lca.register_forward_hook(self._forward_hook)
        self.target_model.SA_lca.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations["lca"] = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients["lca"] = grad_output[0]

    def generate(self, x, target_fn):
        self.model.eval()
        self.model.zero_grad()

        rPPG = self.model(x)
        target = target_fn(rPPG)
        target.backward()

        acts = self.activations["lca"]
        grads = self.gradients["lca"]

        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)
        cam /= cam.max() + 1e-6

        return cam
# ============== Save Grad-CAM heatmaps as images ==================
