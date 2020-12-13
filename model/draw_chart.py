import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4,5))
ax = fig.subplots()
ax.set_xticks([0, 1])
ax.set_xticklabels(['audio researchers', 'non-technical subjects'])
ours = [61, 63.75]
gansynth = [39, 36.25]
ax1 = ax.bar(range(2), gansynth, width=0.5)
ax2 = ax.bar(range(2), ours, bottom=gansynth, width=0.5)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(['GANSynth', 'ours'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.grid(True, axis='y', color='black', alpha=0.2, linestyle=':')

for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "{} %".format(h1), ha="center", va="center")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "{} %".format(h2), ha="center", va="center")

plt.savefig('./paper/assets/figures/quality_result.pdf')
