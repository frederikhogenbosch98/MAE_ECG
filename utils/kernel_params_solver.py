def find_padding(stride, L_in, kernel_size):
    padding = ((stride - 1)* L_in - stride + kernel_size) / 2
    return padding




L_in = 150
kernel_size = 7
stride = 1

padding = find_padding(stride=stride, L_in=L_in, kernel_size=kernel_size)

print(padding)