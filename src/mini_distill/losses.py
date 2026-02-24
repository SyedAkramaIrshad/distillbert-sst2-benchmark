import torch
import torch.nn.functional as F


def compute_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """KL-divergence distillation loss with temperature scaling."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    return kd_loss


def compute_total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, ce_loss, kd_loss)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")

    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = compute_kd_loss(student_logits, teacher_logits, temperature=temperature)
    total = alpha * kd_loss + (1 - alpha) * ce_loss
    return total, ce_loss, kd_loss
