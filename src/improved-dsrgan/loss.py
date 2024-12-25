import torch
import torch.nn.functional as F
import torch.nn as nn
from haar_pytorch import HaarForward

"""
The DDPM-PA loss function [1] is:

    L = Lsimple + λ1 * Lvlb + λ2 * Limg + λ3 * Lhf + λ4 * Lhfmse

where

    Lsimple = MSE loss between predicted and actual noise
    Lvlb    = Loss outlined in "Improved Denoising Diffusion Probabilistic Models" paper [2]
    Limg    = Pairwise similarity between baseline and finetuned images
    Lhf     = Pairwise similarity between high frequency Haar wavelet features of baseline and finetuned images 
    Lhfmse  = MSE loss between high frequency Haar wavelet features of baseline and finetuned images
    λ1...4  = Scalar weightings, see [1] for exact values

References:
[1] https://arxiv.org/abs/2211.03264
[2] https://arxiv.org/abs/2102.09672
"""

def predict_x_0(noise_pred, noisy_image, alphas_cumprod, timesteps):
    """
    Given a noisy image and the predicted noise component of that image we will make a prediction 
    for the clean, un-noised image using:

      x₀ = (1/√(αₜ)) * xₜ − (√(1 − αₜ) / √(αₜ)) * ε₀(xₜ, t)
    
    """
    a_bar = alphas_cumprod[timesteps]
    x_0_pred = (
        (1 / torch.sqrt(a_bar.view(4, 1, 1, 1))) * noisy_image
        - ( torch.sqrt(1 - a_bar.view(4, 1, 1, 1)) / torch.sqrt(a_bar.view(4, 1, 1, 1)) ) * noise_pred
    )
    return x_0_pred

def kl_divergence(p, q):
    """
      Calculate KL divergence with the two categorical distributions p and q:

        sum_over_x∈X( p(x) * log( p(x) / q(x) ))

    """

    entropy_list = [p[x] * torch.log2(p[x] / q[x]) for x in range(len(p))]
    return sum(entropy_list)

def loss_img(noise_pred_ada, noise_pred_sou, noisy_images, alphas_cumprod, timesteps):
    """
    Pairwise similarity loss for generated images:
     
        sum_over_i( Dₖₗ( pᵢᵃᵈᵃ || pᵢˢᵒᵘ ) )

    """
    # Get a prediction for the x₀ for both the source and adapted model
    x_0_ada = predict_x_0(noise_pred_ada, noisy_images, alphas_cumprod, timesteps)
    x_0_sou = predict_x_0(noise_pred_sou, noisy_images, alphas_cumprod, timesteps)

    # Get every combination of two lists of indexes where i ≠ j
    combinations_i = []
    combinations_j = []
    for i in range(noisy_images.shape[0]):
        for j in range(noisy_images.shape[0]):
            if i != j:
                combinations_i.append(i)
                combinations_j.append(j)
    idx_i = torch.tensor(combinations_i)
    idx_j = torch.tensor(combinations_j)

    # Use combinations arrays to create selection multiple permutations of x₀
    x_0_ada_i = x_0_ada[idx_i].flatten(1, 3)
    x_0_ada_j = x_0_ada[idx_j].flatten(1, 3)
    x_0_sou_i = x_0_sou[idx_i].flatten(1, 3)
    x_0_sou_j = x_0_sou[idx_j].flatten(1, 3)

    # Calculate cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_sim_ada = cos(x_0_ada_i, x_0_ada_j)
    cosine_sim_sou = cos(x_0_sou_i, x_0_sou_j)

    # Calculate softmax
    softmax = nn.Softmax()
    p_ada = softmax(cosine_sim_ada)
    p_sou = softmax(cosine_sim_sou)

    # Calculate KL divergence
    kl = kl_divergence(p_ada, p_sou)
    return kl


def loss_hf(noise_pred_ada, noise_pred_sou, noisy_images, alphas_cumprod, timesteps):
    """
    Pairwise similarity loss for high frequency components of Haar wavelet:
     
        sum_over_i( Dₖₗ( pfᵢᵃᵈᵃ || pfᵢˢᵒᵘ ) )

    """
    # Get a prediction for the x₀ for both the source and adapted model
    x_0_ada = predict_x_0(noise_pred_ada, noisy_images, alphas_cumprod, timesteps)
    x_0_sou = predict_x_0(noise_pred_sou, noisy_images, alphas_cumprod, timesteps)

    # Haar wavelet transform
    haar = HaarForward()
    wavelets_ada = haar(x_0_ada)
    wavelets_sou = haar(x_0_sou)
    c = noisy_images.shape[1] # channels (typically 3 i.e. RGD)
    # hf = LH + HL + HH
    hf_ada = wavelets_ada[:,c:c*2,:,:] + wavelets_ada[:,c*2:c*3,:,:] + wavelets_ada[:,c*3:c*4,:,:]
    hf_sou = wavelets_sou[:,c:c*2,:,:] + wavelets_sou[:,c*2:c*3,:,:] + wavelets_sou[:,c*3:c*4,:,:]

    # Get every combination of two lists of indexes where i ≠ j
    combinations_i = []
    combinations_j = []
    for i in range(noisy_images.shape[0]):
        for j in range(noisy_images.shape[0]):
            if i != j:
                combinations_i.append(i)
                combinations_j.append(j)
    idx_i = torch.tensor(combinations_i)
    idx_j = torch.tensor(combinations_j)

    # Use combinations arrays to create selection multiple permutations of x₀
    hf_ada_i = hf_ada[idx_i].flatten(1, 3)
    hf_ada_j = hf_ada[idx_j].flatten(1, 3)
    hf_sou_i = hf_sou[idx_i].flatten(1, 3)
    hf_sou_j = hf_sou[idx_j].flatten(1, 3)

    # Calculate cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_sim_ada = cos(hf_ada_i, hf_ada_j)
    cosine_sim_sou = cos(hf_sou_i, hf_sou_j)

    # Calculate softmax
    softmax = nn.Softmax()
    p_ada = softmax(cosine_sim_ada)
    p_sou = softmax(cosine_sim_sou)

    # Calculate KL divergence
    kl = kl_divergence(p_ada, p_sou)
    return kl

def loss_hfmse(noise_pred_ada, noisy_images, clean_images, alphas_cumprod, timesteps):
    """
    Mean squared error between Haar transform of predicted image and actual image

        ||hf (x̄₀) − hf (x₀)||²

    """
    # Get a prediction for the x₀ for both the source and adapted model
    x_0_pred = predict_x_0(noise_pred_ada, noisy_images, alphas_cumprod, timesteps)

    # Haar wavelet transform
    haar = HaarForward()
    wavelets_pred = haar(x_0_pred)
    wavelets_clean = haar(clean_images)
    c = noisy_images.shape[1] # channels (typically 3 i.e. RGD)
    # hf = LH + HL + HH
    hf_pred = wavelets_pred[:,c:c*2,:,:] + wavelets_pred[:,c*2:c*3,:,:] + wavelets_pred[:,c*3:c*4,:,:]
    hf_clean = wavelets_clean[:,c:c*2,:,:] + wavelets_clean[:,c*2:c*3,:,:] + wavelets_clean[:,c*3:c*4,:,:]

    # Mean squared error between high frequency component of the wavelet transform
    return F.mse_loss(hf_pred, hf_clean)
