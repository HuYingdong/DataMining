import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=20)

x_coords = np.linspace(-100, 100, 1000)
y_coords = np.linspace(-100, 100, 1000)
points = []
for y in y_coords:
    for x in x_coords:
        if ((x*0.03)**2+(y*0.03)**2-1)**3-(x*0.03)**2*(y*0.03)**3 <= 0:
            points.append({"x": x, "y": y})

heart_x = list(map(lambda point: point["x"], points))
heart_y = list(map(lambda point: point["y"], points))

# cmap="cool" "magma" "rainbow" "Reds" "spring" "viridis" "gist_rainbow"
plt.scatter(heart_x, heart_y, s=10, alpha=0.5, c=range(len(heart_x)), cmap="rainbow")
plt.title('送你一颗心', color='red', fontproperties=myfont)
words = '小心心'
plt.text(x=(np.mean(heart_x)-len(words)), y=np.mean(heart_y), s=words, color='red', fontproperties=myfont)
plt.show()

