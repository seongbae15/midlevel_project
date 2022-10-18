"""
# select font from list
import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
font_list

# delete cache before rerun [ /Users/b-09/.matplotlib/fontlist-v330.json ]
# matplotlib.get_cachedir()   # 'C:\\Users\\Plateau\\.matplotlib'


# Usage:
import numpy as np
import matplotlib.pyplot as plt
import _hangul_font_for_pyplot
t = np.arange(0, 10, 0.01)
plt.figure()
plt.title("삼각함수")
plt.plot(t, np.sin(t))
plt.show()
"""

import matplotlib
from matplotlib import pyplot, font_manager, rc
import platform

if platform.system() == "Darwin":
    font_path = "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf"
elif platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    print("windows")

font_name = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rc("font", family=font_name)
matplotlib.pyplot.rcParams["axes.unicode_minus"] = False

if platform.system() == "Darwin":
    print("Font AppleMyungjo.ttf is set!")
elif platform.system() == "Windows":
    print("Font NanumGothic.ttf is set!")
