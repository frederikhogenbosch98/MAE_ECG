import re
import matplotlib.pyplot as plt
import numpy as np
# data_text = input('give me text')

# # Data text as provided
data_text = """
epoch 1: average loss: 0.4062277, val loss: 0.2336779, duration: 10.25s, lr: 0.000010
epoch 2: average loss: 0.2191824, val loss: 0.2164868, duration: 9.12s, lr: 0.000132
epoch 3: average loss: 0.1627874, val loss: 0.1202355, duration: 9.15s, lr: 0.000255
epoch 4: average loss: 0.0866653, val loss: 0.0460951, duration: 9.17s, lr: 0.000378
epoch 5: average loss: 0.0391050, val loss: 0.0252185, duration: 9.21s, lr: 0.000500
epoch 6: average loss: 0.0223989, val loss: 0.0159508, duration: 9.20s, lr: 0.000498
epoch 7: average loss: 0.0149569, val loss: 0.0108447, duration: 9.13s, lr: 0.000500
epoch 8: average loss: 0.0114459, val loss: 0.0100765, duration: 9.25s, lr: 0.000499
epoch 9: average loss: 0.0092150, val loss: 0.1546705, duration: 9.25s, lr: 0.000493
epoch 10: average loss: 0.0267414, val loss: 0.0123050, duration: 9.32s, lr: 0.000484
epoch 11: average loss: 0.0098632, val loss: 0.0099913, duration: 9.28s, lr: 0.000472
epoch 12: average loss: 0.0084896, val loss: 0.0088094, duration: 9.19s, lr: 0.000456
epoch 13: average loss: 0.0092346, val loss: 0.0087259, duration: 9.60s, lr: 0.000436
epoch 14: average loss: 0.0081484, val loss: 0.0073560, duration: 9.30s, lr: 0.000413
epoch 15: average loss: 0.0070614, val loss: 0.0067048, duration: 9.28s, lr: 0.000387
epoch 16: average loss: 0.0064590, val loss: 0.0066162, duration: 9.28s, lr: 0.000358
epoch 17: average loss: 0.0061594, val loss: 0.0058369, duration: 9.19s, lr: 0.000327
epoch 18: average loss: 0.0059153, val loss: 0.0057847, duration: 9.18s, lr: 0.000293
epoch 19: average loss: 0.0057651, val loss: 0.0055850, duration: 9.24s, lr: 0.000256
epoch 20: average loss: 0.0056367, val loss: 0.0053697, duration: 9.28s, lr: 0.000218
epoch 21: average loss: 0.0055745, val loss: 0.0059069, duration: 9.32s, lr: 0.000179
epoch 22: average loss: 0.0055091, val loss: 0.0052519, duration: 9.22s, lr: 0.000138
epoch 23: average loss: 0.0055840, val loss: 0.0053484, duration: 9.20s, lr: 0.000096
epoch 24: average loss: 0.0054108, val loss: 0.0051631, duration: 9.22s, lr: 0.000053
epoch 25: average loss: 0.0053172, val loss: 0.0052262, duration: 9.26s, lr: 0.000010
epoch 26: average loss: 0.0052994, val loss: 0.0055898, duration: 9.28s, lr: 0.000254
epoch 27: average loss: 0.0053468, val loss: 0.0058862, duration: 9.26s, lr: 0.000255
epoch 28: average loss: 0.0053105, val loss: 0.0052486, duration: 9.21s, lr: 0.000254
epoch 29: average loss: 0.0051232, val loss: 0.0068210, duration: 9.20s, lr: 0.000252
epoch 30: average loss: 0.0053521, val loss: 0.0051268, duration: 9.26s, lr: 0.000247
epoch 31: average loss: 0.0049969, val loss: 0.0048832, duration: 9.24s, lr: 0.000241
epoch 32: average loss: 0.0054044, val loss: 0.0489986, duration: 9.36s, lr: 0.000233
epoch 33: average loss: 0.0147212, val loss: 0.0087250, duration: 9.48s, lr: 0.000223
epoch 34: average loss: 0.0086312, val loss: 0.0070684, duration: 9.33s, lr: 0.000212
epoch 35: average loss: 0.0071554, val loss: 0.0064606, duration: 9.16s, lr: 0.000199
epoch 36: average loss: 0.0065642, val loss: 0.0059187, duration: 9.22s, lr: 0.000184
epoch 37: average loss: 0.0062576, val loss: 0.0057919, duration: 9.40s, lr: 0.000168
epoch 38: average loss: 0.0061291, val loss: 0.0057121, duration: 9.22s, lr: 0.000151
epoch 39: average loss: 0.0058684, val loss: 0.0164664, duration: 9.14s, lr: 0.000133
epoch 40: average loss: 0.0073165, val loss: 0.0054503, duration: 9.18s, lr: 0.000114
epoch 41: average loss: 0.0058662, val loss: 0.0055400, duration: 9.21s, lr: 0.000094
epoch 42: average loss: 0.0056542, val loss: 0.0053822, duration: 9.21s, lr: 0.000074
epoch 43: average loss: 0.0055214, val loss: 0.0055671, duration: 9.13s, lr: 0.000053
epoch 44: average loss: 0.0054993, val loss: 0.0056485, duration: 9.10s, lr: 0.000031
epoch 45: average loss: 0.0054308, val loss: 0.0053741, duration: 9.13s, lr: 0.000010
epoch 46: average loss: 0.0054097, val loss: 0.0052741, duration: 9.17s, lr: 0.000132
epoch 47: average loss: 0.0054557, val loss: 0.0055087, duration: 9.16s, lr: 0.000132
epoch 48: average loss: 0.0055111, val loss: 0.0053906, duration: 9.15s, lr: 0.000132
epoch 49: average loss: 0.0052687, val loss: 0.0054214, duration: 9.08s, lr: 0.000131
epoch 50: average loss: 0.0051423, val loss: 0.0051591, duration: 9.15s, lr: 0.000129
epoch 51: average loss: 0.0051492, val loss: 0.0049056, duration: 9.19s, lr: 0.000125
epoch 52: average loss: 0.0049712, val loss: 0.0047958, duration: 9.23s, lr: 0.000121
epoch 53: average loss: 0.0049199, val loss: 0.0051823, duration: 9.23s, lr: 0.000117
epoch 54: average loss: 0.0048675, val loss: 0.0050294, duration: 9.08s, lr: 0.000111
epoch 55: average loss: 0.0047617, val loss: 0.0046719, duration: 9.13s, lr: 0.000104
epoch 56: average loss: 0.0047189, val loss: 0.0045052, duration: 9.16s, lr: 0.000097
epoch 57: average loss: 0.0046410, val loss: 0.0045541, duration: 9.18s, lr: 0.000089
epoch 58: average loss: 0.0046123, val loss: 0.0049977, duration: 9.24s, lr: 0.000081
epoch 59: average loss: 0.0045573, val loss: 0.0045880, duration: 9.16s, lr: 0.000072
epoch 60: average loss: 0.0045496, val loss: 0.0045550, duration: 9.18s, lr: 0.000062
epoch 61: average loss: 0.0044835, val loss: 0.0045891, duration: 9.85s, lr: 0.000052
epoch 62: average loss: 0.0046329, val loss: 0.0047229, duration: 9.39s, lr: 0.000042
epoch 63: average loss: 0.0044223, val loss: 0.0042718, duration: 9.32s, lr: 0.000031
epoch 64: average loss: 0.0044039, val loss: 0.0044569, duration: 9.25s, lr: 0.000021
epoch 65: average loss: 0.0044100, val loss: 0.0045210, duration: 9.33s, lr: 0.000010
epoch 66: average loss: 0.0043839, val loss: 0.0043734, duration: 9.30s, lr: 0.000071
epoch 67: average loss: 0.0043664, val loss: 0.0043250, duration: 9.37s, lr: 0.000071
epoch 68: average loss: 0.0043563, val loss: 0.0043662, duration: 9.30s, lr: 0.000071
epoch 69: average loss: 0.0045154, val loss: 0.0044401, duration: 9.29s, lr: 0.000070
epoch 70: average loss: 0.0042817, val loss: 0.0042153, duration: 9.16s, lr: 0.000069
epoch 71: average loss: 0.0042300, val loss: 0.0043646, duration: 9.70s, lr: 0.000068
epoch 72: average loss: 0.0042073, val loss: 0.0044554, duration: 9.73s, lr: 0.000066
epoch 73: average loss: 0.0041531, val loss: 0.0041255, duration: 9.34s, lr: 0.000063
epoch 74: average loss: 0.0041671, val loss: 0.0039945, duration: 9.31s, lr: 0.000060
epoch 75: average loss: 0.0041042, val loss: 0.0043544, duration: 9.21s, lr: 0.000057
epoch 76: average loss: 0.0040463, val loss: 0.0040199, duration: 9.21s, lr: 0.000054
epoch 77: average loss: 0.0040027, val loss: 0.0042331, duration: 9.28s, lr: 0.000050
epoch 78: average loss: 0.0039825, val loss: 0.0040489, duration: 10.10s, lr: 0.000045
epoch 79: average loss: 0.0039706, val loss: 0.0044423, duration: 9.62s, lr: 0.000041
epoch 80: average loss: 0.0041147, val loss: 0.0040601, duration: 9.40s, lr: 0.000036
epoch 81: average loss: 0.0039174, val loss: 0.0040026, duration: 9.44s, lr: 0.000031
epoch 82: average loss: 0.0038969, val loss: 0.0040304, duration: 9.43s, lr: 0.000026
epoch 83: average loss: 0.0038899, val loss: 0.0039710, duration: 9.34s, lr: 0.000021
epoch 84: average loss: 0.0038784, val loss: 0.0040007, duration: 9.59s, lr: 0.000015
epoch 85: average loss: 0.0038591, val loss: 0.0037109, duration: 9.63s, lr: 0.000010
epoch 86: average loss: 0.0038518, val loss: 0.0040874, duration: 9.62s, lr: 0.000040
epoch 87: average loss: 0.0038425, val loss: 0.0039872, duration: 9.58s, lr: 0.000041
epoch 88: average loss: 0.0038369, val loss: 0.0038214, duration: 9.53s, lr: 0.000041
epoch 89: average loss: 0.0038574, val loss: 0.0044688, duration: 9.61s, lr: 0.000040
epoch 90: average loss: 0.0038460, val loss: 0.0037152, duration: 9.90s, lr: 0.000040
epoch 91: average loss: 0.0037861, val loss: 0.0037059, duration: 9.60s, lr: 0.000039
epoch 92: average loss: 0.0037626, val loss: 0.0040109, duration: 9.31s, lr: 0.000038
epoch 93: average loss: 0.0037314, val loss: 0.0037426, duration: 9.36s, lr: 0.000037
epoch 94: average loss: 0.0037132, val loss: 0.0038181, duration: 9.36s, lr: 0.000035
epoch 95: average loss: 0.0037016, val loss: 0.0037314, duration: 9.59s, lr: 0.000034
epoch 96: average loss: 0.0036951, val loss: 0.0036988, duration: 9.38s, lr: 0.000032
epoch 97: average loss: 0.0036838, val loss: 0.0040747, duration: 9.46s, lr: 0.000030
epoch 98: average loss: 0.0036928, val loss: 0.0037955, duration: 9.29s, lr: 0.000028
epoch 99: average loss: 0.0036302, val loss: 0.0038583, duration: 9.32s, lr: 0.000025
epoch 100: average loss: 0.0036530, val loss: 0.0036171, duration: 9.32s, lr: 0.000023
epoch 101: average loss: 0.0036077, val loss: 0.0036920, duration: 9.35s, lr: 0.000021
epoch 102: average loss: 0.0036014, val loss: 0.0037463, duration: 9.21s, lr: 0.000018
epoch 103: average loss: 0.0035783, val loss: 0.0036970, duration: 9.34s, lr: 0.000015
epoch 104: average loss: 0.0035621, val loss: 0.0035858, duration: 9.34s, lr: 0.000013
epoch 105: average loss: 0.0035615, val loss: 0.0040781, duration: 9.30s, lr: 0.000010
epoch 106: average loss: 0.0035564, val loss: 0.0036806, duration: 9.36s, lr: 0.000025
epoch 107: average loss: 0.0035734, val loss: 0.0036402, duration: 9.19s, lr: 0.000025
epoch 108: average loss: 0.0035610, val loss: 0.0038012, duration: 9.21s, lr: 0.000025
epoch 109: average loss: 0.0035485, val loss: 0.0040184, duration: 9.35s, lr: 0.000025
epoch 110: average loss: 0.0035363, val loss: 0.0039717, duration: 9.46s, lr: 0.000025
epoch 111: average loss: 0.0035475, val loss: 0.0038676, duration: 9.52s, lr: 0.000024
epoch 112: average loss: 0.0035225, val loss: 0.0035772, duration: 9.65s, lr: 0.000024
epoch 113: average loss: 0.0034900, val loss: 0.0034535, duration: 9.46s, lr: 0.000023
epoch 114: average loss: 0.0034759, val loss: 0.0034732, duration: 9.22s, lr: 0.000023
epoch 115: average loss: 0.0034668, val loss: 0.0033725, duration: 9.34s, lr: 0.000022
epoch 116: average loss: 0.0034732, val loss: 0.0037188, duration: 9.38s, lr: 0.000021
epoch 117: average loss: 0.0034725, val loss: 0.0036459, duration: 9.41s, lr: 0.000020
epoch 118: average loss: 0.0034485, val loss: 0.0034540, duration: 9.91s, lr: 0.000019
epoch 119: average loss: 0.0034277, val loss: 0.0034943, duration: 9.69s, lr: 0.000018
epoch 120: average loss: 0.0034161, val loss: 0.0037126, duration: 9.44s, lr: 0.000017
epoch 121: average loss: 0.0034323, val loss: 0.0040414, duration: 9.63s, lr: 0.000015
epoch 122: average loss: 0.0034434, val loss: 0.0035419, duration: 9.33s, lr: 0.000014
epoch 123: average loss: 0.0034006, val loss: 0.0033391, duration: 9.48s, lr: 0.000013
epoch 124: average loss: 0.0033964, val loss: 0.0035053, duration: 9.65s, lr: 0.000011
epoch 125: average loss: 0.0034156, val loss: 0.0034925, duration: 9.52s, lr: 0.000010
epoch 126: average loss: 0.0034018, val loss: 0.0034126, duration: 9.55s, lr: 0.000018
epoch 127: average loss: 0.0033801, val loss: 0.0035197, duration: 9.42s, lr: 0.000018
epoch 128: average loss: 0.0033798, val loss: 0.0036290, duration: 9.44s, lr: 0.000018
epoch 129: average loss: 0.0033947, val loss: 0.0034325, duration: 9.33s, lr: 0.000018
epoch 130: average loss: 0.0034625, val loss: 0.0033146, duration: 9.27s, lr: 0.000017
epoch 131: average loss: 0.0033634, val loss: 0.0032601, duration: 9.24s, lr: 0.000017
epoch 132: average loss: 0.0033669, val loss: 0.0033231, duration: 9.30s, lr: 0.000017
epoch 133: average loss: 0.0033397, val loss: 0.0036483, duration: 9.21s, lr: 0.000017
epoch 134: average loss: 0.0033473, val loss: 0.0033124, duration: 9.18s, lr: 0.000016
epoch 135: average loss: 0.0033311, val loss: 0.0033310, duration: 9.21s, lr: 0.000016
epoch 136: average loss: 0.0033255, val loss: 0.0035049, duration: 9.23s, lr: 0.000015
epoch 137: average loss: 0.0033215, val loss: 0.0033549, duration: 9.29s, lr: 0.000015
epoch 138: average loss: 0.0033400, val loss: 0.0037647, duration: 9.23s, lr: 0.000014
epoch 139: average loss: 0.0033621, val loss: 0.0033320, duration: 9.15s, lr: 0.000014
epoch 140: average loss: 0.0033465, val loss: 0.0034994, duration: 9.23s, lr: 0.000013
epoch 141: average loss: 0.0033089, val loss: 0.0034027, duration: 9.27s, lr: 0.000013
epoch 142: average loss: 0.0033140, val loss: 0.0033827, duration: 9.26s, lr: 0.000012
epoch 143: average loss: 0.0033069, val loss: 0.0033654, duration: 9.31s, lr: 0.000011
epoch 144: average loss: 0.0032918, val loss: 0.0032741, duration: 9.14s, lr: 0.000011
epoch 145: average loss: 0.0032869, val loss: 0.0034955, duration: 9.18s, lr: 0.000010
epoch 146: average loss: 0.0032959, val loss: 0.0033830, duration: 9.26s, lr: 0.000014
epoch 147: average loss: 0.0033164, val loss: 0.0033945, duration: 9.32s, lr: 0.000014
epoch 148: average loss: 0.0033058, val loss: 0.0034139, duration: 9.35s, lr: 0.000014
epoch 149: average loss: 0.0032660, val loss: 0.0034827, duration: 9.19s, lr: 0.000014
epoch 150: average loss: 0.0032628, val loss: 0.0032913, duration: 9.18s, lr: 0.000014
epoch 151: average loss: 0.0032541, val loss: 0.0033957, duration: 9.20s, lr: 0.000014
epoch 152: average loss: 0.0032610, val loss: 0.0033200, duration: 9.27s, lr: 0.000013
epoch 153: average loss: 0.0032564, val loss: 0.0035138, duration: 9.27s, lr: 0.000013
epoch 154: average loss: 0.0032572, val loss: 0.0032517, duration: 9.22s, lr: 0.000013
epoch 155: average loss: 0.0032393, val loss: 0.0033684, duration: 9.16s, lr: 0.000013
epoch 156: average loss: 0.0032228, val loss: 0.0033082, duration: 9.21s, lr: 0.000013
epoch 157: average loss: 0.0032420, val loss: 0.0033247, duration: 9.26s, lr: 0.000012
epoch 158: average loss: 0.0032501, val loss: 0.0034251, duration: 9.28s, lr: 0.000012
epoch 159: average loss: 0.0032460, val loss: 0.0031841, duration: 9.26s, lr: 0.000012
epoch 160: average loss: 0.0032329, val loss: 0.0034839, duration: 9.16s, lr: 0.000012
epoch 161: average loss: 0.0032177, val loss: 0.0033118, duration: 9.20s, lr: 0.000011
epoch 162: average loss: 0.0032292, val loss: 0.0031911, duration: 9.54s, lr: 0.000011
epoch 163: average loss: 0.0032188, val loss: 0.0035410, duration: 9.75s, lr: 0.000011
epoch 164: average loss: 0.0032155, val loss: 0.0033888, duration: 9.43s, lr: 0.000010
epoch 165: average loss: 0.0032085, val loss: 0.0033648, duration: 9.15s, lr: 0.000010
epoch 166: average loss: 0.0031994, val loss: 0.0033944, duration: 9.21s, lr: 0.000012
epoch 167: average loss: 0.0032068, val loss: 0.0032531, duration: 9.23s, lr: 0.000012
epoch 168: average loss: 0.0031907, val loss: 0.0034926, duration: 9.32s, lr: 0.000012
epoch 169: average loss: 0.0031921, val loss: 0.0034432, duration: 9.29s, lr: 0.000012
epoch 170: average loss: 0.0031947, val loss: 0.0034538, duration: 9.18s, lr: 0.000012
epoch 171: average loss: 0.0031829, val loss: 0.0035634, duration: 9.15s, lr: 0.000012
epoch 172: average loss: 0.0031988, val loss: 0.0038488, duration: 9.18s, lr: 0.000012
epoch 173: average loss: 0.0031964, val loss: 0.0033426, duration: 9.21s, lr: 0.000012
epoch 174: average loss: 0.0031835, val loss: 0.0032979, duration: 9.23s, lr: 0.000012
epoch 175: average loss: 0.0031994, val loss: 0.0031836, duration: 9.17s, lr: 0.000011
epoch 176: average loss: 0.0031776, val loss: 0.0031366, duration: 9.12s, lr: 0.000011
epoch 177: average loss: 0.0031851, val loss: 0.0034146, duration: 9.13s, lr: 0.000011
epoch 178: average loss: 0.0031909, val loss: 0.0034624, duration: 9.19s, lr: 0.000011
epoch 179: average loss: 0.0031557, val loss: 0.0032718, duration: 9.21s, lr: 0.000011
epoch 180: average loss: 0.0031650, val loss: 0.0033507, duration: 9.21s, lr: 0.000011
epoch 181: average loss: 0.0031490, val loss: 0.0031113, duration: 9.10s, lr: 0.000011
epoch 182: average loss: 0.0031555, val loss: 0.0031435, duration: 9.14s, lr: 0.000010
epoch 183: average loss: 0.0031463, val loss: 0.0035856, duration: 9.32s, lr: 0.000010
epoch 184: average loss: 0.0031632, val loss: 0.0036101, duration: 9.43s, lr: 0.000010
epoch 185: average loss: 0.0031377, val loss: 0.0032235, duration: 9.32s, lr: 0.000010
epoch 186: average loss: 0.0031377, val loss: 0.0033689, duration: 9.17s, lr: 0.000011
epoch 187: average loss: 0.0031347, val loss: 0.0035700, duration: 9.17s, lr: 0.000011
epoch 188: average loss: 0.0031271, val loss: 0.0032163, duration: 9.24s, lr: 0.000011
epoch 189: average loss: 0.0031285, val loss: 0.0029894, duration: 9.29s, lr: 0.000011
epoch 190: average loss: 0.0031277, val loss: 0.0033542, duration: 9.29s, lr: 0.000011
epoch 191: average loss: 0.0031341, val loss: 0.0037371, duration: 9.22s, lr: 0.000011
epoch 192: average loss: 0.0031128, val loss: 0.0030757, duration: 9.21s, lr: 0.000011
epoch 193: average loss: 0.0031287, val loss: 0.0033690, duration: 9.26s, lr: 0.000011
epoch 194: average loss: 0.0031165, val loss: 0.0032930, duration: 9.28s, lr: 0.000011
epoch 195: average loss: 0.0031134, val loss: 0.0031170, duration: 9.29s, lr: 0.000011
epoch 196: average loss: 0.0031059, val loss: 0.0033343, duration: 9.27s, lr: 0.000011
epoch 197: average loss: 0.0031188, val loss: 0.0033525, duration: 9.19s, lr: 0.000011
epoch 198: average loss: 0.0031186, val loss: 0.0032553, duration: 9.22s, lr: 0.000011
epoch 199: average loss: 0.0031087, val loss: 0.0031137, duration: 9.27s, lr: 0.000010
epoch 200: average loss: 0.0030890, val loss: 0.0034253, duration: 9.31s, lr: 0.000010
epoch 201: average loss: 0.0030955, val loss: 0.0033800, duration: 9.32s, lr: 0.000010
epoch 202: average loss: 0.0030842, val loss: 0.0035284, duration: 9.16s, lr: 0.000010
epoch 203: average loss: 0.0030969, val loss: 0.0031745, duration: 9.21s, lr: 0.000010
epoch 204: average loss: 0.0030835, val loss: 0.0032143, duration: 9.25s, lr: 0.000010
epoch 205: average loss: 0.0030810, val loss: 0.0035141, duration: 9.29s, lr: 0.000010
epoch 206: average loss: 0.0030747, val loss: 0.0030994, duration: 9.35s, lr: 0.000010
epoch 207: average loss: 0.0030678, val loss: 0.0032748, duration: 9.21s, lr: 0.000010
epoch 208: average loss: 0.0030784, val loss: 0.0031201, duration: 9.19s, lr: 0.000010
epoch 209: average loss: 0.0030649, val loss: 0.0033441, duration: 9.22s, lr: 0.000010
epoch 210: average loss: 0.0030812, val loss: 0.0032641, duration: 9.24s, lr: 0.000010
epoch 211: average loss: 0.0030793, val loss: 0.0032585, duration: 9.28s, lr: 0.000010
epoch 212: average loss: 0.0030840, val loss: 0.0033784, duration: 9.24s, lr: 0.000010
epoch 213: average loss: 0.0031289, val loss: 0.0030281, duration: 9.17s, lr: 0.000010
epoch 214: average loss: 0.0032363, val loss: 0.0032120, duration: 9.21s, lr: 0.000010
epoch 215: average loss: 0.0030941, val loss: 0.0037997, duration: 9.23s, lr: 0.000010
epoch 216: average loss: 0.0030758, val loss: 0.0030656, duration: 9.28s, lr: 0.000010
epoch 217: average loss: 0.0030570, val loss: 0.0034751, duration: 9.24s, lr: 0.000010
epoch 218: average loss: 0.0030605, val loss: 0.0030041, duration: 9.11s, lr: 0.000010
epoch 219: average loss: 0.0030671, val loss: 0.0030355, duration: 9.15s, lr: 0.000010
epoch 220: average loss: 0.0030806, val loss: 0.0035144, duration: 9.20s, lr: 0.000010
epoch 221: average loss: 0.0030657, val loss: 0.0032961, duration: 9.25s, lr: 0.000010
epoch 222: average loss: 0.0030395, val loss: 0.0030270, duration: 9.28s, lr: 0.000010
epoch 223: average loss: 0.0030587, val loss: 0.0036091, duration: 9.15s, lr: 0.000010
epoch 224: average loss: 0.0030478, val loss: 0.0032347, duration: 9.16s, lr: 0.000010
epoch 225: average loss: 0.0030425, val loss: 0.0032339, duration: 9.17s, lr: 0.000010
epoch 226: average loss: 0.0030387, val loss: 0.0030947, duration: 9.26s, lr: 0.000010
epoch 227: average loss: 0.0030326, val loss: 0.0031401, duration: 9.26s, lr: 0.000010
epoch 228: average loss: 0.0030394, val loss: 0.0034572, duration: 9.13s, lr: 0.000010
epoch 229: average loss: 0.0030282, val loss: 0.0031273, duration: 9.12s, lr: 0.000010
epoch 230: average loss: 0.0030322, val loss: 0.0030897, duration: 9.17s, lr: 0.000010
epoch 231: average loss: 0.0030233, val loss: 0.0032813, duration: 9.26s, lr: 0.000010
epoch 232: average loss: 0.0030268, val loss: 0.0031747, duration: 9.26s, lr: 0.000010
epoch 233: average loss: 0.0030356, val loss: 0.0032263, duration: 9.20s, lr: 0.000010
epoch 234: average loss: 0.0030227, val loss: 0.0035120, duration: 9.11s, lr: 0.000010
epoch 235: average loss: 0.0030060, val loss: 0.0035321, duration: 9.11s, lr: 0.000010
epoch 236: average loss: 0.0030174, val loss: 0.0031802, duration: 9.18s, lr: 0.000010
epoch 237: average loss: 0.0030113, val loss: 0.0029491, duration: 9.29s, lr: 0.000010
epoch 238: average loss: 0.0030094, val loss: 0.0032775, duration: 9.23s, lr: 0.000010
epoch 239: average loss: 0.0030132, val loss: 0.0039270, duration: 9.28s, lr: 0.000010
epoch 240: average loss: 0.0030166, val loss: 0.0031234, duration: 9.49s, lr: 0.000010
epoch 241: average loss: 0.0030145, val loss: 0.0031732, duration: 9.32s, lr: 0.000010
epoch 242: average loss: 0.0030109, val loss: 0.0030586, duration: 9.62s, lr: 0.000010
epoch 243: average loss: 0.0030211, val loss: 0.0032159, duration: 9.53s, lr: 0.000010
epoch 244: average loss: 0.0030196, val loss: 0.0034436, duration: 9.30s, lr: 0.000010
epoch 245: average loss: 0.0030172, val loss: 0.0035849, duration: 9.25s, lr: 0.000010
epoch 246: average loss: 0.0030010, val loss: 0.0030484, duration: 9.27s, lr: 0.000010
epoch 247: average loss: 0.0030010, val loss: 0.0033792, duration: 9.28s, lr: 0.000010
epoch 248: average loss: 0.0029813, val loss: 0.0031215, duration: 9.35s, lr: 0.000010
epoch 249: average loss: 0.0030070, val loss: 0.0030051, duration: 9.25s, lr: 0.000010
epoch 250: average loss: 0.0030095, val loss: 0.0032988, duration: 9.17s, lr: 0.000010
epoch 251: average loss: 0.0030276, val loss: 0.0032216, duration: 9.23s, lr: 0.000010
epoch 252: average loss: 0.0029996, val loss: 0.0036158, duration: 9.46s, lr: 0.000010
epoch 253: average loss: 0.0029800, val loss: 0.0034271, duration: 9.29s, lr: 0.000010
epoch 254: average loss: 0.0029704, val loss: 0.0030088, duration: 9.46s, lr: 0.000010
epoch 255: average loss: 0.0029848, val loss: 0.0033528, duration: 9.19s, lr: 0.000010
"""

# Regex pattern to find average and val loss
pattern = r"epoch \d+: average loss: (\d+\.\d+), val loss: (\d+\.\d+)"

# Find all matches
matches = re.findall(pattern, data_text)
avg_losses = []
val_losses = []
# Output results
for i, (avg_loss, val_loss) in enumerate(matches, start=1):
    avg_losses.append(float(avg_loss))
    val_losses.append(float(val_loss))

    # print(f"Epoch {i}: Average Loss = {avg_loss}, Validation Loss = {val_loss}")


epochs = np.arange(1,256)
plt.plot(epochs, avg_losses, label='training loss')
plt.plot(epochs, val_losses, label='validation loss')
plt.title('losses for 250 epoch run')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
