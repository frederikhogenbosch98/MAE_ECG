out_channels = int(input("out channels: "))
in_channels = int(input("in channels: "))
kernel_size = int(input("kernel size: "))

print(f'number of params: {out_channels * (in_channels * kernel_size**2 + 1)}')
