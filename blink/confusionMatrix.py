from le.metrics import accuracy_score, confusion_matrix
import seaborn as sns

from keras.models import Model, load_model

from train import start_time, x_val, y_val

model = load_model('models/%s.h5' % (start_time))

y_pred = model.predict(x_val/255.)
y_pred_logical = (y_pred > 0.5).astype(int)

print('test acc: %s' % accuracy_score(y_val, y_pred_logical))
cm = confusion_matrix(y_val, y_pred_logical)
sns.heatmap(cm, annot=True)