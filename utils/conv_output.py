input_size = input("input size: ")
kernel_size = input("kernel size: ")
stride = input("stride: ")
padding = input("padding: ")

output = ((int(input_size)+(2*int(padding))-int(kernel_size))/int(stride))+1
print(output)