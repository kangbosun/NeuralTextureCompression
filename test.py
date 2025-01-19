import torch
from piqa import SSIM

# 예시 텐서 (N, C, H, W)
x = torch.rand(3, 3, 64, 64)
y = torch.rand(3, 3, 64, 64)

criterion = SSIM()
loss = 1 - criterion(x, y)  # SSIM이 높을수록 '손실'을 낮추려는 경우
print("1 - SSIM:", loss.item())