import torch
import os
# number of missed frames of all videos (torch.Size([4, 14])) , 4 views and 14 actions
miss_f = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/HrNet/missed_frame_05.pt')
#Number of missed poses in four views , torch.Size([4, 14, 17]), 4 views, 14 actions and 17 poses
miss_p = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/HrNet/missed_posses_05.pt')
# Number of total frames in 4 views , torch.Size([4, 14]), 4 views and 14 actions
frame_t = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/HrNet/total_frame_num_05.pt')
print(frame_t.shape)
# Number of unmissed poses, torch.Size([4, 14]), 4 views and 14 actions
correct_p = torch.load('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/HrNet/total_correct_poses_05.pt')

import numpy as np
import torch
import matplotlib.pyplot as plt

N = 14
ind = np.arange(N)
width = 0.23
width_t = 0.05
ind_t = ind - 0.10
plt.rcParams["figure.figsize"] = (28,10)
fontsize = 20
fig, ax = plt.subplots()

#90353b
#55752f
#1a476f
#e37e00
#1B1919FF
xvals = frame_t[0,:]
print(xvals)
xvals = xvals.numpy()
bar1_1 = plt.bar(ind_t, xvals, width_t, color = '#1B1919FF')

for i,p in enumerate(bar1_1):
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+200,
      s="100\n%",
      ha='center')

xvals = miss_f[0,:]
xvals = xvals.numpy()
bar1 = plt.bar(ind, xvals, width, color = '#90353b')

for i,p in enumerate(bar1):
   height_t = bar1_1[i].get_height()
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+200,
      s="{:.1f}\n%".format((height/height_t)*100),
      ha='center')

xvals = frame_t[1,:]
xvals = xvals.numpy()
# bar1_1 = plt.bar(ind_t, xvals, width_t, color = '#1B1919FF')

yvals = miss_f[1,:]
yvals = yvals.numpy()
bar2 = plt.bar(ind+width, yvals, width, color='#55752f')

for i,p in enumerate(bar2):
   height_t = bar1_1[i].get_height()
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+200,
      s="{:.1f}\n%".format((height/height_t)*100),
      ha='center')

xvals = frame_t[2,:]
xvals = xvals.numpy()
# bar1_1 = plt.bar(ind_t, xvals, width_t, color = '#1B1919FF')

zvals = miss_f[2,:]
zvals = zvals.numpy()
bar3 = plt.bar(ind+width*2, zvals, width, color = '#1a476f')

for i,p in enumerate(bar3):
   height_t = bar1_1[i].get_height()
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+200,
      s="{:.1f}\n%".format((height/height_t)*100),
      ha='center')

xvals = frame_t[3,:]
xvals = xvals.numpy()
# bar1_1 = plt.bar(ind_t, xvals, width_t, color = '#1B1919FF')

qvals = miss_f[3,:]
qvals = qvals.numpy()
print(frame_t)
print(qvals)
bar4 = plt.bar(ind+width*3, qvals, width, color = '#e37e00')

for i,p in enumerate(bar4):
   height_t = bar1_1[i].get_height()
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+220,
      s="{:.1f}\n%".format((height/height_t)*100),
      ha='center')

plt.xlabel("Actions",fontdict={'fontsize': fontsize})
plt.ylabel('Number of Frames in each action and camera view',fontdict={'fontsize': fontsize})
plt.grid(axis='y')
# plt.title("Missed Frames of RH-HAR Skeleton Dataset")

plt.xticks(ind+width,['Bending', 'SittingDown', 'ClosingCan', 'Reaching', 'Walking', 'Drinking', 'StairsClimbingUp',
            'StairsClimbingDown', 'StandingUp', 'OpeningCan', 'CarryingObject', 'Cleaning', 'PuttingDownObjects',
              'LiftingObject'])
plt.legend( (bar1_1,bar1, bar2, bar3, bar4), ('Total_Frames','Robot View', 'Back View', 'Front View', 'Omni View') )
plt.savefig('/home/reza/PycharmProjects/RHM-HAR-SK-Dataset/HrNet/allviews_t.png')
plt.show()



