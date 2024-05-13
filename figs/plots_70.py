import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

s_random=[100,100,60,80,120,40,40,60,100,100]
s_random_grid=[80,80,80,80,80,80,80,80,80,80]
s_AIDE=[46,46,46,46,46,46,66,46,46,46]

m_random=[80,80,80,40,100,80,40,60,40,40]
m_random_grid=[80,80,80,80,80,80,80,80,80,80]
m_AIDE=[46,46,46,46,46,46,46,46,46,46]

l_random=[40,80,20,60,20,40,40,40,80,120]
l_random_grid=[80,80,80,80,80,80,80,80,80,80]
l_AIDE=[46,46,46,46,46,46,46,46,46,46]

random=[sum(l_random)/len(l_random),sum(m_random)/len(m_random), sum(s_random)/len(s_random)]
random_grid=[sum(l_random_grid)/len(l_random_grid),sum(m_random_grid)/len(m_random_grid), sum(s_random_grid)/len(s_random_grid)]
AIDE=[sum(l_AIDE)/len(l_AIDE),sum(m_AIDE)/len(m_AIDE),sum(s_AIDE)/len(s_AIDE)]

colors = sns.color_palette("mako")
colors=[colors[i] for i in [1,3,5]]

width = 0.2  # the width of the bars
ind=np.asarray([1,2,3])
capsize=4

fig, ax = plt.subplots(figsize=(8, 3.5))
# rects1 = ax.bar(ind - 1.1*width, AIDE, width, edgecolor='k',yerr=[[min(l_AIDE),min(m_AIDE),min(s_AIDE)],[max(l_AIDE),max(m_AIDE),max(s_AIDE)]],
#                 label='AIDE', color=colors[0], capsize=capsize)
# rects2 = ax.bar(ind , random, width, edgecolor='k',yerr=[[min(l_random)+random[0],min(m_random)+random[1],min(s_random)+random[2]],[max(l_random)-random[0],max(m_random)-random[1],max(s_random)-random[2]]],
#                 label='Random', color=colors[1], capsize=capsize)
# rects1 = ax.bar(ind +1.1*width, random_grid, width, edgecolor='k',yerr=[[min(l_random_grid),min(m_random_grid),min(s_random_grid)],[max(l_random_grid),max(m_random_grid),max(s_random_grid)]],
#                 label='Random Grid', color=colors[2], capsize=capsize)

rects1 = ax.bar(ind - 1.1*width, AIDE, width, edgecolor='k',yerr=[np.std(l_AIDE), np.std(m_AIDE),np.std(s_AIDE)],
                label='AIDE', color=colors[0], capsize=capsize)
rects2 = ax.bar(ind , random, width, edgecolor='k',yerr=[np.std(l_random),np.std(m_random),np.std(s_random)],
                label='Random', color=colors[1], capsize=capsize)
rects3 = ax.bar(ind +1.1*width, random_grid, width, edgecolor='k',yerr=[np.std(l_random_grid),np.std(m_random_grid),np.std(s_random_grid)],
                label='Random Grid', color=colors[2], capsize=capsize)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Samples')
ax.set_xlabel('F-Measure (%)')
ax.set_ylim([0,160])
# ax.set_title('')
ax.set_xticks(ind)
ax.set_xticklabels(('Large', 'Medium', 'Small'))
ax.legend()
fig.tight_layout()
# plt.show()

# plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5,8.5, 9.5], [0, 1, 2, 3, 4, 5, 6,7,8,9])
# plt.plot([-0.5,0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5, 8.5],nU, color=colors[1], linewidth=3, alpha=0.8)
# plt.plot([-0.5,0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5, 8.5],nG, color=colors[2], linewidth=3, alpha=0.8)
# plt.plot([-0.5,0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5, 8.5],nall, color=colors[0], linewidth=3, alpha=0.8)

# plt.legend()
plt.savefig(f"n_samples_70.png")
