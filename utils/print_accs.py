import matplotlib.pyplot as plt

R_LIST = [5, 10, 15, 20, 25, 30, 35, 40]
cpd_half = [0.6719738924050633, 0.9100079113924051, 0.9315664556962026, 0.9390822784810127, 0.9366099683544303, 0.9333465189873418, 0.9375988924050633, 0.9404667721518988]
tucker_half = [0.7814477848101266, 0.921875, 0.9439280063291139, 0.9495648734177216, 0.9485759493670886, 0.9335443037974683, 0.9421479430379747, 0.9396756329113924]
cpd_full = [0.11244066455696203, 0.11244066455696203, 0.860067246835443, 0.8774723101265823, 0.8511669303797469, 0.8622428797468354, 0.8472112341772152, 0.8711431962025317]

plt.plot(R_LIST, cpd_half, label='cpd')
plt.plot(R_LIST, tucker_half, label='tucker')
plt.axhline(y=0.9218, color='r', linestyle='-', label='uncompressed')
plt.title('Accuracies at various ranks for encoder cpd and tucker')
plt.xlabel('Ranks')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(R_LIST, cpd_full, label='cpd')
plt.axhline(y=0.9218, color='r', linestyle='-', label='uncompressed')
plt.title('Accuracies at various ranks for full cpd')
plt.xlabel('Ranks')
plt.ylabel('accuracy')
plt.legend()
plt.show()