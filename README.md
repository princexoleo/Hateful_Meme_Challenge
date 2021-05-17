# Hateful_Meme_Challenge
Hateful Memes dataset contains real hate speech. The Real Hateful Memes dataset consists of more than 10,000 newly created examples by Facebook AI. 

Hateful Memes dataset contains real hate speech. The Real Hateful Memes dataset consists of more
than 10,000 newly created examples by Facebook AI.
Solutions: As the memes contain images with text, it becomes related to computer vision and natural
language processing problems.The target is to classify the memes in either hateful or not hateful class, so
I have to analyze the images, texts which is why I chose multimodal architecture for hateful memes
classification. For multimodal architecture I chose the ResNet-152(vision), fastText(language) model.
Comparing their results. I believe that multimodality holds the key to problems with as much variety as
natural language and understanding computer vision evaluation, and embodied AI.
Brief description of sample code:
First, for deep analysis in the dataset, I have to know some statistics of different objects that are present
in thoses images. To find out, I applied an object detection deep learning model called “Detectron 2”. In
order to that model, I installed detectron2 libraries and imported others libraries. After importing libraries, I
created a detectron2 config and a detectron2 DefaultPredictor to run inference on images. Then I loaded the
weights/pretrained model in “DefaultPredictor'' that comes from “from detectron2 import model_zoo”.
Then I went through my dataset and applied DefaultPredictor for prediction classes, and collected those
classes found in images into an object list.. Then I filtered the total object found in a single image and the
total unique number objects found in a single image and stored them in a pandas dataframe. Later I
visualized them into different formats by using matplotlib libraries and also tried to find the different
categories' answers from the dataframe, for example, which images contained above 5 unique objects
and how many total images were there that contained most objects in a single image. You can find those
code in this link: [Data Analysis of Hateful Memes]

## Dataset PreProcessing: 
The dataset contains 3 files and 1 directory. The directory is called
img and it contains all the meme images, where I have access: train, dev and test. The image
files were named <id>.png, where id is a unique 5 digit number. Each remaining jsonl file
contains id, img, text and label. I loaded them by using “pd.read_json(filepath, lines=True)”. I
also made a HatefulMemesDataset class which extended the torch.utils.data.Dataset class. In
this class, it takes data_path, img_dir, image_transformation, text_transformation, balance,dev_limit, ranodm_state as an input argument. With this class I could load the dataset into a
pytorch dataloader like trainloader, valloader. [source code]

## Vision Module : 
In the vision module, I used the ResNet-152 model. The ResNet-152 was
imported from torchvision libraries with pretrained weights. I extracted the features from the
images rather than classifying to overwrite the last layer with an identity transformation. Then I
reduced the dimension using a linear layer and the resnet output was 2048.

```vision_module=torchvision.models.resnet152(pretrained=True)
vision_module.fc = torch.nn.Linear(in_features=2048, out_features=self.vision_feature_dim)
```


## Language Module:
It’s a task to extract features from text. And for extracting these features I used fastText libraries’
model.FastText is an open-source, free, lightweight library that allows users to learn text representations
and text classifiers.
```
language_transform = fasttext.train_unsupervised(
str(ft_path),
model= self.hparams.get("fasttext_model", "cbow"),
dim=self.embedding_dim
)
```
I passed the outputs of our text to transform through an additional trainable layer rather than fine-tuning
the transform. For that I added another linear layer.
```
language_module = torch.nn.Linear(
in_features=self.embedding_dim,
out_features=self.language_feature_dim
)
```
## The Late Fusion:
There are many fusions available in the research work. Among which I chose the later fusion
method. That means the feature extracted from the vision and language module will be fusioned
and passed through a linear layer.
```
self.fusion = torch.nn.Linear(
in_features=(language_feature_dim + vision_feature_dim),
out_features=fusion_output_size
)
```
## The model forward method:
This method receives a text and an image. Firstly it passes the text through our language
module and the output comes from a language module which will be passed through an
activation function called relu. Like this, the image will be passed through our vision module and
the output from a vision module will also be passed through relu activation function. Then I
combined the text feature and image features and applied the late fusion method. After applying
late fusion, I passed the output through a fully connected linear layer. The input feature size of
the linear layer is the output size of fusion and the output feature size is equal to the number of
classes. For finding the class’ probability distribution I used a softmax activation function. Thenthe prediction was passed through a loss function. Finally, this method returns the prediction
and loss value.
```
def forward(self, text, image, label=None):
  text_features = torch.nn.functional.relu(
  self.language_module(text)
  )
  image_features = torch.nn.functional.relu(
  self.vision_module(image)
  )
  combined = torch.cat(
  [text_features, image_features], dim=1
  )
  fused = self.dropout(
  torch.nn.functional.relu(
  self.fusion(combined)
  )
 )
  logits = self.fc(fused)
  pred = torch.nn.functional.softmax(logits)
  loss = (
  self.loss_fn(pred, label)
  if label is not None else label
  )
  return (pred, loss)

```

The rest of the code is done using PyTorch Lightning libraries which are basically based on the PyTorch
framework. And the other parameter that I used for solving this problem is given below.
```
hparams = {
  "train_path": train_path,
  "dev_path": dev_path,
  "img_dir": data_dir,
  # Optional params
  "embedding_dim": 150,
  "language_feature_dim": 300,
  "vision_feature_dim": 300,
  "fusion_output_size": 256,
  "output_path": "model-outputs",
  "dev_limit": None,
  "lr": 0.00005,
  "max_epochs": 10,
  "n_gpu": 1,
  "batch_size": 4,
  # allows us to "simulate" having larger batches
  "accumulate_grad_batches": 16,
  "early_stop_patience": 3,
}
```

You can find the original source code of the project in the following link :
https://github.com/princexoleo/Hateful_Meme_Challenge
