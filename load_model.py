import rain_prediction as rp
import joblib

new_rain_pred = joblib.load("rain_prediction.joblib")
print(new_rain_pred["model"])

test_pred2 = new_rain_pred["model"].predict(rp.X_test)
new_accuracy = rp.accuracy_score(rp.test_targets, test_pred2)
print(new_accuracy)

