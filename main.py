import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

data_x = [(7.2, 2.5), (6.4, 2.2), (6.3, 1.5), (7.7, 2.2), (6.2, 1.8), (5.7, 1.3), (7.1, 2.1), (5.8, 2.4), (5.2, 1.4), (5.9, 1.5), (7.0, 1.4), (6.8, 2.1), (7.2, 1.6), (6.7, 2.4), (6.0, 1.5), (5.1, 1.1), (6.6, 1.3), (6.1, 1.4), (6.7, 2.1), (6.4, 1.8), (5.6, 1.3), (6.9, 2.3), (6.4, 1.9), (6.9, 2.3), (6.5, 2.2),
           (6.0, 1.5), (5.6, 1.1), (5.6, 1.5), (6.0, 1.0), (6.0, 1.8), (6.7, 2.5), (7.7, 2.3), (5.5, 1.1), (5.8, 1.0), (6.9, 2.1), (6.6, 1.4), (6.3, 1.6), (6.1, 1.4), (5.0, 1.0), (7.7, 2.0), (4.9, 1.7), (7.2, 1.8), (6.8, 1.4), (6.1, 1.2), (5.8, 1.9), (6.3, 2.5), (5.7, 2.0), (6.5, 1.8), (7.6, 2.1), (6.3, 1.5),
           (6.7, 1.4), (6.4, 2.3), (6.2, 2.3), (6.3, 1.9), (5.5, 1.3), (7.9, 2.0), (6.7, 1.8), (6.4, 1.3), (6.5, 2.0), (6.5, 1.5), (6.9, 1.5), (5.6, 1.3), (5.8, 1.2), (6.7, 2.3), (6.0, 1.6), (5.7, 1.2), (5.7, 1.0), (5.5, 1.0), (6.1, 1.4), (6.3, 1.8), (5.7, 1.3), (6.1, 1.3), (5.5, 1.3), (6.3, 1.3), (5.9, 1.8),
           (7.7, 2.3), (6.5, 2.0), (5.6, 2.0), (6.7, 1.7), (5.7, 1.3), (5.5, 1.2), (5.0, 1.0), (5.8, 1.9), (6.2, 1.3), (6.2, 1.5), (6.3, 2.4), (6.4, 1.5), (7.4, 1.9), (6.8, 2.3), (5.6, 1.3), (5.8, 1.2), (7.3, 1.8), (6.7, 1.5), (6.3, 1.8), (6.0, 1.6), (6.4, 2.1), (6.1, 1.8), (5.9, 1.8), (5.4, 1.5), (4.9, 1.0)]

data_y = [1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1,
           1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1]

clf = SVC(kernel='linear')
clf.fit(data_x, data_y)
v = clf.support_vectors_

lin_clf = LinearSVC()
lin_clf.fit(data_x, data_y)
pred_y = lin_clf.predict(data_x)

wrong_class = sum(data_y != pred_y)
wrong_class_percent = wrong_class / len(data_y) * 100

print("Число неверных классификаций:", wrong_class)
print("Процент неверных классификаций:", wrong_class_percent)

w = lin_clf.coef_[0]
b = lin_clf.intercept_[0]
data_x = np.array(data_x)
data_y = tuple(data_y)
line_x = range(min(round(x[0])-1 for x in data_x), max(round(x[0])+1 for x in data_x))
line_y = -(w[0] / w[1]) * line_x - (b / w[1])


def sign_color(i):
    if i == 1:
        return 'red'
    if i == -1:
        return 'blue'
def setting_sign_color(data):
    p = list()
    for y in data:
        p.append(sign_color(y))
    return p

def sign_marker(y):
    if y == 1:
        return 'o'
    if y == -1:
        return 's'

plt.scatter(data_x[:, 0], data_x[:, 1], color=setting_sign_color(data_y), marker='o')
plt.scatter(v[:, 0], v[:, 1], s=1, marker='s', color='black')
plt.plot(line_x, line_y, color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([3, 9])
plt.ylim([0, 3])
plt.show()

