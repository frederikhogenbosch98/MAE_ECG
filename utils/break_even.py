import numpy as np
import matplotlib.pyplot as plt

# Define the variables
R = 100  # R ranges from 0 to 199
S = [1, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 128, 128, 64, 64]  # example value for S
H = [128, 128, 64, 64, 32, 32, 16, 16, 16, 16, 32, 32, 64, 64, 128, 128]  # example value for H
W = [128, 128, 64, 64, 32, 32, 16, 16, 16, 16, 32, 32, 64, 64, 128, 128]  # example value for W
D = 3  # example value for D
T = [64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 128, 128, 64, 64, 1]  # example value for T

# Calculate each term
def  FLOPs_sr(R, S, H, W):
    return 4 * R * S * H * W
def  FLOPs_w(R, H, W):
    return  4 * D * H * W * R**2
def  FLOPs_h(R, H, W):
    return  4 * D * H * W * R**2
def  FLOPs_rt(R, H, W, T):
    return  4 * R * T * H * W
def FLOPs_og(S, T, H, W):
    return 4 * S * D**2 * T * H * W


S = 64
T = 128
D = 3
H = 64
W = 64
R = 100

print(FLOPs_og(S, T, H, W))
print(FLOPs_sr(R, S, H, W) + FLOPs_w(R, H, W) + FLOPs_h(R, H, W) + FLOPs_rt(R, H, W, T))

# epoch_durations = [294, 294, 296, 307, 314, 325, 345, 378, 413, 454, 505, 568]

# # Sum the terms
# total_FLOPs = []
# total_FLOPs_og = []
# for i in range(8):
#     sr = FLOPs_sr(R, S[i], H[i], W[i])
#     h = FLOPs_w(R, H[i], W[i])
#     w = FLOPs_h(R, H[i], W[i])
#     rt = FLOPs_rt(R, H[i], W[i], T[i])
#     total_FLOPs.append(sr + h + w + rt)
#     total_FLOPs_og.append(FLOPs_og(S[i], T[i], H[i], W[i]))


# # 2HW T SD2,

# # Calculate the original FLOPs


# sum_cp = np.sum(total_FLOPs)
# sum_un = np.sum(total_FLOPs_og)

# print(sum_un)
# print(sum_cp)

# print(sum_cp/sum_un)



# # Plot the total FLOPs as a function of R
# plt.plot(R, total_FLOPs, label='Total FLOPs')
# plt.plot(R, epoch_durations, label='Total FLOPs')
# plt.axhline(FLOPs_og, color='r', linestyle='--', label='Original FLOPs')

# # Add labels and legend
# plt.xlabel('R')
# plt.ylabel('FLOPs')
# plt.title('Total FLOPs as a function of R')
# plt.legend()

# # Show the plot
# plt.show()