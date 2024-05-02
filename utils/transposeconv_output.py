input_size = input("Input size: ")
kernel_size = input("Kernel size: ")
stride = input("Stride: ")
padding = input("Padding: ")
dilation = input("Dilation: ")
output_padding = input("Output padding: ")

output = ((int(input_size) - 1) * int(stride) - 2 * int(padding) + int(dilation) * (int(kernel_size) - 1) + int(output_padding) + 1)
print("Output size:", output)
