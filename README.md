# This is the source code of my perfessional course ,i.e. AI experiment 3 at UESTC(2023 SUMMER). 
You can git clone it into your local environment, execute it, and hopefully you may get some take home message. Enjoy yourself, and have fun！

(by the way... don't forget to star this project if you find it useful, that do means a lot to me!! salute)
# One thing notable:
You may need a GPU to run the code,casue it's a time-consuming work for CPU...
# Some learning materials and resource are down below:
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)
# Reproduction Tutorial
## 1) First,Git clone this respository to your local laptop:

```
git clone https://github.com/Levi-Ackman/AI_EXP_3.git
```

## 2) Then,create the environment needed by : 
In pip (Advised):

```
pip install -r requirements.txt
```
## 3)Run the code by a simple line : 
 Modify 'data_path' and 'json_path' in parser.add_argument('...') in run.py to the paths of your own TaxiData and json files. As follows:
 <p float="left">
  <img src="图片一.jpg?raw=true" width="100%" />
  </p>
  
## 4)Run the code by a simple line : 
In Linux (Advised):

```
python run.py
```
## 5) results and visulization
a. After running, you can find the training results and indicators of the corresponding model in the test_dict folder.
b. In the test_results folder, you can get the prediction curve of the corresponding model and the real value comparison chart.
c. Under the checkpoint folder, you can get the training weight file of the corresponding model.

