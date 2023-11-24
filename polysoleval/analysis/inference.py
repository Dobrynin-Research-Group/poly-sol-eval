import asyncio

import torch

from polysoleval.range import Range


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
        b_range (psst.Range): The Range of the parameter to be evaluated.

    Returns:
        float: The inferred value of the parameter.
    """
    with torch.no_grad():
        pred: torch.Tensor = model(visc_normed.unsqueeze_(0))

    b_range.unnormalize(pred)

    return pred.squeeze().item()


async def do_inferences(
    bg_model: torch.nn.Module,
    visc_normed_bg: torch.Tensor,
    bg_range: Range,
    bth_model: torch.nn.Module,
    visc_normed_bth: torch.Tensor,
    bth_range: Range,
) -> tuple[float, float]:
    """Perform two inferences concurrently to obtain the estimates for Bg and Bth.

    Args:
        bg_model (torch.nn.Module): Model to infer the Bg value.
        visc_normed_bg (torch.Tensor): The normalized viscosity tensor for inferring
          Bg.
        bg_range (psst.Range): The range of Bg values to assist in normalization.
        bth_model (torch.nn.Module): Model to infer the Bth value.
        visc_normed_bth (torch.Tensor): The normalized viscosity tensor for inferring
          Bth.
        bth_range (psst.Range): The range of Bth values to assist in normalization.

    Returns:
        tuple[float, float]: The estimated values of Bg and Bth, respectively.
    """
    bg_task = asyncio.create_task(inference_model(bg_model, visc_normed_bg, bg_range))
    bth_task = asyncio.create_task(
        inference_model(bth_model, visc_normed_bth, bth_range)
    )
    bg = await bg_task
    bth = await bth_task
    return bg, bth
