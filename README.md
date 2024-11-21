### Environment Setup
```
python -m venv .env
.env/Script/activate
```
### Installation
```
pip install -r requirements.txt
```
### Train MMAction on custom dataset
Download pre-trained checkpoint at [MMAction2 Model Zoo](https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html)
```
python train.py
```
### Inference
```
python main.py
```