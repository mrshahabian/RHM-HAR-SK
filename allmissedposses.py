import torch
import os
# number of missed frames of all videos (torch.Size([4, 14])) , 4 views and 14 actions
miss_f = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/Yolo7/missed_frame_05.pt')
#Number of missed poses in four views , torch.Size([4, 14, 17]), 4 views, 14 actions and 17 poses
miss_p = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/Yolo7/missed_posses_05.pt')
# Number of total frames in 4 views , torch.Size([4, 14]), 4 views and 14 actions
frame_t = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/Yolo7/total_frame_num_05.pt')
print(frame_t.shape)
# Number of unmissed poses, torch.Size([4, 14]), 4 views and 14 actions
correct_p = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/Yolo7/total_correct_poses_05.pt')

import numpy as np
import torch
import matplotlib.pyplot as plt

N = 17
ind = np.arange(N)
width = 0.25
fontsize = 20
width_t = 0.05
plt.rcParams["figure.figsize"] = (26, 10)
fig, ax = plt.subplots()
# 90353b
# 55752f
# 1a476f

xvals = torch.sum(miss_p[0, :, :], dim=0)

pos_t = torch.sum(correct_p[0, :, :], dim=0)
height_t = xvals + pos_t
height_t = height_t.numpy()
# xvals = xvals.numpy()
xvals = (xvals.numpy() / height_t) * 100
bar1 = plt.bar(ind, xvals, width, color='#90353b')

# bar1_1 = plt.bar(ind, pos_t, width_t, color = 'y')

# for i, p in enumerate(bar1):
#     #  height_t = xvals + pos_t
#     height = p.get_height()
#     ax.text(x=p.get_x() + p.get_width() / 2, y=height + .10,
#             s="{:.1f}\n%".format((height / height_t[i]) * 100),
#             ha='center')

yvals = torch.sum(miss_p[1, :, :], dim=0)
pos_t = torch.sum(correct_p[1, :, :], dim=0)
height_t2 = yvals + pos_t
height_t2 = height_t2.numpy()
# yvals = yvals.numpy()
yvals = (yvals.numpy() / height_t2) * 100
bar2 = plt.bar(ind + width, yvals, width, color='#55752f')

# bar1_1 = plt.bar(ind+width, pos_t, width_t, color = 'y')

# for i, p in enumerate(bar2):
#     #  height_t = xvals + pos_t
#     height = p.get_height()
#     ax.text(x=p.get_x() + p.get_width() / 2, y=height + .10,
#             s="{:.1f}\n%".format((height / height_t2[i]) * 100),
#             ha='center')

zvals = torch.sum(miss_p[2, :, :], dim=0)
pos_t = torch.sum(correct_p[2, :, :], dim=0)
height_t3 = zvals + pos_t
height_t3 = height_t3.numpy()
# zvals = zvals.numpy()
zvals = (zvals.numpy() / height_t3) * 100
bar3 = plt.bar(ind + width * 2, zvals, width, color='#1a476f')


# bar1_1 = plt.bar(ind+width*2, pos_t, width_t, color = 'y')

# for i, p in enumerate(bar3):
#     #  height_t = xvals + pos_t
#     height = p.get_height()
#     ax.text(x=p.get_x() + p.get_width() / 2, y=height + .10,
#             s="{:.1f}\n%".format((height / height_t3[i]) * 100),
#             ha='center')

plt.xlabel("Skeleton Poses", fontdict={'fontsize': fontsize})
plt.ylabel('Percentage of missed poses', fontdict={'fontsize': fontsize})
plt.grid()
# plt.title("Average Missed Poses of all Actions of RH-HAR Skeleton Dataset")

plt.xticks(ind + width, ['Nose', 'Left-eye', 'Right-eye', 'Left-ear', 'Right-ear',
                         'Left-shoulder', 'Right-shoulder', 'Left-elbow', 'Right-elbow',
                         'Left-wrist', 'Right-wrist', 'Left-Hip', 'Right-hip',
                         'Left-knee', 'Right-knee', 'Left-ankle', 'Right-ankle'])
plt.legend((bar1, bar2, bar3), ('Robot View', 'Back View', 'Front View'))
plt.savefig('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/Yolo7/pos-avg_all.png')
plt.show()
