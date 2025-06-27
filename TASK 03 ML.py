import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set correct path and folder names
BASE_DIR = "C:/Users/Msi/Downloads/archive/train"
CATEGORIES = ["cats", "dogs"]
IMG_SIZE = 64
data = []

for category in CATEGORIES:
    folder_path = os.path.join(BASE_DIR, category)
    label = CATEGORIES.index(category)  # 0 for cats, 1 for dogs

    for img_name in os.listdir(folder_path)[:500]:  # Load 500 from each
        try:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append([img.flatten(), label])
        except Exception as e:
            print(f"Skipped {img_name}: {e}")

if len(data) == 0:
    raise Exception(" No images loaded. Check the path and folder names.")

random.shuffle(data)
X, y = zip(*data)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Show sample predictions
for i in range(5):
    index = random.randint(0, len(X_test) - 1)
    img = X_test[index].reshape(IMG_SIZE, IMG_SIZE)
    predicted = y_pred[index]
    actual = y_test[index]

    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {'Dog' if predicted else 'Cat'} | Actual: {'Dog' if actual else 'Cat'}")
    plt.axis('off')
    plt.show()


