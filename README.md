# NBA Playoff Prediction

**NBA Playoff Prediction** is a machine learning project designed to predict how far an NBA team is likely to progress in the playoffs based on its regular season performance.

The model outputs the probabilities for a team to reach the following stages:
- 🏀 Conference Semi-Finals  
- 🏀 Conference Finals  
- 🏀 NBA Finals  
- 🏆 NBA Champion

This project uses a **Chain Classifier** composed of **XGBoost** models to predict these outcomes.

---
## 🧠 Model Inputs

The model takes the following inputs for each team:
- Conference (East or West)
- Season record (wins/losses)
- Conference seed
- Overall NBA seed
- Offensive rating rank
- Defensive rating rank
- Result from two seasons ago
- Result from the last season

---

## 🚀 Inference

### 🔗 Use via Streamlit App  
For the easiest experience, use the interactive web application:  
👉 [NBA Playoff Predictor on Streamlit](https://nbaplayoff-prediction.streamlit.app/)

### 🐍 Use via Python
#### Run inference
python -m Inference.inference

---

## ⚙️ Training the Model
Training code is located in the Training/ folder. To retrain the model:
python -m Training.training

---

## 📊 Dataset

The dataset includes **all NBA teams that qualified for the playoffs** from the **1983–1984 season to the 2023–2024 season**, including teams that entered through the **play-in tournament**.

For each team, the dataset provides:
- **Model input features**:  
  - Conference  
  - Season record  
  - Conference seed  
  - Overall NBA seed  
  - Offensive rating rank  
  - Defensive rating rank  
  - Playoff result two seasons ago  
  - Playoff result from the previous season  
- **Label**: The actual playoff result of the team (Conference Semi-Finalist, Finalist, NBA Finalist, NBA Champion)

---

## 🧪 Running Tests
To run unit tests:
pytest tests/

---

## 🤝 Contributing
Contributions are welcome!
Feel free to open issues, suggest improvements, or submit pull requests.

---

## 📄 License
This project is licensed under the MIT License.

---

## 📬 Contact
Questions or suggestions?
Reach out at: adridevolder@hotmail.com

