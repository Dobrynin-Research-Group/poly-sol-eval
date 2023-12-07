from asyncio import create_task, gather

import torch
from polysoleval.exceptions import PSSTException

from polysoleval.models import Range


async def inference_model(
    model: torch.nn.Module,
    visc_normed: torch.Tensor,
    b_range: Range,
) -> float:
    """Run the processed viscosity data through a pre-trained model to gain the
    inference results for either :math:`B_g` or :math:`B_{th}`.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model.
        visc_normed (torch.Tensor): The normalized viscosity, appropriate to the
          parameter to be evaluated.
        b_range (Range): The Range of the parameter to be evaluated. Used for
          transforming the inferred value from a range of :math:`[0, 1]` to its
          true value.

    Returns:
        float: The inferred value of the parameter.
    """
    with torch.no_grad():
        try:
            pred: torch.Tensor = model(visc_normed.unsqueeze_(0))
        except RuntimeError as re:
            raise PSSTException.InferenceRuntimeError from re

    b_range.unnormalize(pred)

    return pred.item()


async def do_inferences(
    bg_model: torch.nn.Module,
    visc_normed_bg: torch.Tensor,
    bg_range: Range,
    bth_model: torch.nn.Module,
    visc_normed_bth: torch.Tensor,
    bth_range: Range,
) -> tuple[float, float]:
    """Perform two inferences concurrently.

    Args:
        bg_model (torch.nn.Module): Model to infer the :math:`B_g` value.
        visc_normed_bg (torch.Tensor): The normalized viscosity tensor for inferring
          :math:`B_g`.
        bg_range (psst.Range): The range of :math:`B_g` values to assist in
          normalization.
        bth_model (torch.nn.Module): Model to infer the :math:`B_{th}` value.
        visc_normed_bth (torch.Tensor): The normalized viscosity tensor for inferring
          :math:`B_{th}`.
        bth_range (psst.Range): The range of :math:`B_{th}` values to assist in
          normalization.

    Returns:
        tuple[float, float]: The estimated values of :math:`B_g` and :math:`B_{th}`,
          respectively.
    """
    bg_task = create_task(inference_model(bg_model, visc_normed_bg, bg_range))
    bth_task = create_task(inference_model(bth_model, visc_normed_bth, bth_range))
    bg, bth = await gather(bg_task, bth_task)
    return bg, bth
