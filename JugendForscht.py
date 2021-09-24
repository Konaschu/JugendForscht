import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

#
# Weiß zu Schwarz
#

img = Image.open('Capture.png')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((0, 0, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("Capture1.png", "PNG")

#
# In Rot invertieren
#


img2 = cv2.imread('Capture1.png')
rot = img2.copy()
rot[:, :, 0] = 0
rot[:, :, 1] = 0
cv2.imwrite('Capture2.png', rot)
cv2.waitKey(0)


#
# Tabelle durchschnittlichen Farbwerten
#

def visualize_colors(cluster, centroids):
    # Histogram erstellung
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Cluster erstellen in abhängigkeit der Farbe
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50),
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect


# Bild zu Pixeln
image = cv2.imread('Capture2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))

# Menge der angezeigten Farben +1 (Schwarz)
cluster = KMeans(n_clusters=10).fit(reshape)

# Tabelle erstellen und als Bild anzeigen
visualize = visualize_colors(cluster, cluster.cluster_centers_)
visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
cv2.imwrite('Capture3.png', visualize)
cv2.waitKey()

#
# Schwarz aus Tabelle entfernen
#

img3 = cv2.imread('Capture3.png')
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)

crop = img3[y:y + h, x:x + w]
cv2.imwrite('Capture4.png', crop)

#
# Durchschnittlichen Farbwert ermitteln
#

img4 = cv2.imread('Capture4.png')
avg_color_per_row = np.average(img4, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)

#
# Auf Tabelle anzeigen
#

img = io.imread('https://i.imgur.com/tAkp45Y.png')
n = int(input("Ergbenis ohne Nachkommastelle: "))
if n > 119:
    n = 119
elif n < 51:
    n = 51
n1 = n*10
zahl = n1-500

plt.figure()
plt.imshow(img)
plt.scatter(zahl, 13, s=50, c='black', marker='o')
plt.show()