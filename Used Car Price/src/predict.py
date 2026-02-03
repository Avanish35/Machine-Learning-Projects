import pickle
import pandas as pd

model = pickle.load(open(r"E:\Projects\Used Car Price\model\car_price_model.pkl", "rb"))

sample_input = {
    "manufacturing_year": 2020,
    "km_driven": 30000,
    "brand": "Hyundai",
    "model": "Creta",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "city": "Delhi",
    "bodytype": "SUV",
    "number_of_owners": 1
}

df = pd.DataFrame([sample_input])
prediction = model.predict(df)

print("Predicted Price:", prediction[0])
