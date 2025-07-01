# Bengali Food Calorie Estimator

![Bengali Food](https://d4t7t8y8xqo0t.cloudfront.net/app/resized/1080X/eazytrendz/3108/trend20210413160402.jpg)

A deep learning-based system that classifies Bengali food images and estimates their calorie content using a MobileNetV2 model.

## Features

- Classifies 20+ common Bengali food items
- Estimates calorie content from food images
- Lightweight MobileNetV2 architecture suitable for deployment
- Easy-to-use command line interface

## Installation

### Prerequisites
- Python 3.8+
- pip
- git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AkibHossainOmi/bengali_food_calorie_estimator
cd bengali_food_calorie_estimator
```

2. Create and activate a virtual environment:

#### On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with your dataset:
```bash
python main.py train --data_dir data/Bengali_Food --epochs 10 --batch_size 32
```

Arguments:
- `--data_dir`: Path to dataset directory (should contain train/val/test subdirectories)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--model_path`: Path to save trained model (default: models/food_classifier.h5)

### Making Predictions

To estimate calories from a food image:
```bash
python main.py predict --image_path "path_to_your_image.jpg"
```

Arguments:
- `--image_path`: Path to food image file
- `--model_path`: Path to trained model (default: models/food_classifier.h5)

Example output:
```
Predicted Food: Biryani
Estimated Calories: 450 kcal
```

## Dataset

The model was trained on a custom dataset containing 20+ Bengali food categories with ~1000 images per category. Each image is labeled with verified calorie information.

Dataset structure:
```
Bengali_Food/
│── biryani/
│── khichuri/
│...other food categories

```

## Model Architecture

The system uses a MobileNetV2 base with custom classification layers:
- Input: 224x224 RGB images
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Custom Head: GlobalAveragePooling2D + Dense layers
- Output: Food class probabilities and calorie regression

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
