HW03Part07 - Cruise Control State Classifier
This repository contains the solution for Section 7: Cruise Control State Classifier from HW03. The goal of this part is to build a binary classifier that predicts whether Adaptive Cruise Control (ACC) is enabled or not from vehicle time-history signals.
Overview
This part was solved in three different versions in order to improve performance by changing the model setup, epochs, feature design, and other training parameters.
The implemented work includes:
preprocessing of decoded CAN signal files
matching front-left wheel speed with ACC status
binary target construction
zero-order hold label alignment
lagged time-history feature construction
train/test split and scaling
training and evaluation of multiple classifier versions
loss and accuracy plots
ONNX model export
inference script
The classifier predicts:
1 if ACC status = 6
0 otherwise
Dataset
Dataset path used:
```bash/data/CPE\\\_487-587/ACCDataset```
For the first implementation, the following files were used from each experiment:
`\\\*\\\_wheel\\\_speed\\\_fl.csv`
matching `\\\*\\\_acc\\\_status.csv`
For the improved versions, additional signals were included:
`\\\*\\\_relative\\\_vel.csv`
`\\\*\\\_lead\\\_distance.csv`
`\\\*\\\_accely.csv`
A total of 13 matched experiments were found and processed.
Data Preparation
The preprocessing pipeline performs the following steps:
Read the `Time` and `Message` columns from each decoded signal file.
Convert front-left wheel speed from km/h to m/s.
Convert ACC status into a binary label:
`1` if status is `6`
`0` otherwise
Remove duplicate ACC timestamps.
Align labels and additional signals to the wheel-speed timeline using zero-order hold.
Construct historical lag features.
Implemented Versions
Version 1
The first version uses only front-left wheel speed history:
`v\\\_t`
`v\\\_t-1`
...
`v\\\_t-10`
This version produced a strong result and in one of the completed runs the accuracy was obtained at about 87%.
Version 2
The second version uses multiple signals and more derived features:
current values of speed, relative velocity, lead distance, and longitudinal acceleration
lagged histories
first-difference features
rolling mean and rolling standard deviation
This version was created to improve the model by using richer information. In another completed run, the obtained accuracy was about 78%.
Version 3
The third version keeps the richer multi-signal setup but uses:
smaller history length
a smaller and more regularized model
different epochs and learning-rate settings
modified training configuration to reduce overfitting
In another completed run, the obtained accuracy was about 83%.
Files
Main files created for this part:
`src/mchnpkg/deepl/acc\\\_module.py`
`src/mchnpkg/deepl/acc\\\_module\\\_v2.py`
`src/mchnpkg/deepl/acc\\\_module\\\_v3.py`
`scripts/acc\\\_impl.py`
`scripts/acc\\\_impl.sh`
`scripts/acc\\\_inference.py`
`scripts/acc\\\_impl\\\_v2.py`
`scripts/acc\\\_impl\\\_v3.py`
Example Run Commands
Version 1
```bashexport PYTHONPATH=srcpython scripts/acc\\\_impl.py \\\\  --data\\\_dir /data/CPE\\\_487-587/ACCDataset \\\\  --output\\\_dir results/acc\\\_test \\\\  --k 10 \\\\  --sample\\\_size 300000 \\\\  --test\\\_size 0.2 \\\\  --epochs 5 \\\\  --batch\\\_size 256 \\\\  --lr 0.001 \\\\  --num\\\_workers 2```
Version 2
```bashexport PYTHONPATH=srcpython scripts/acc\\\_impl\\\_v2.py \\\\  --data\\\_dir /data/CPE\\\_487-587/ACCDataset \\\\  --output\\\_dir results/acc\\\_v2\\\_test \\\\  --k 10 \\\\  --sample\\\_size 300000 \\\\  --test\\\_ratio 0.2 \\\\  --epochs 10 \\\\  --batch\\\_size 256 \\\\  --lr 0.001 \\\\  --num\\\_workers 2```
Version 3
```bashexport PYTHONPATH=srcpython scripts/acc\\\_impl\\\_v3.py \\\\  --data\\\_dir /data/CPE\\\_487-587/ACCDataset \\\\  --output\\\_dir results/acc\\\_v3\\\_test \\\\  --k 5 \\\\  --sample\\\_size 300000 \\\\  --test\\\_ratio 0.2 \\\\  --epochs 10 \\\\  --batch\\\_size 256 \\\\  --lr 0.0005 \\\\  --num\\\_workers 2```
Inference
After training, inference can be run using the ONNX model from version 1:
```bashexport PYTHONPATH=srcpython scripts/acc\\\_inference.py \\\\  --onnx\\\_model results/acc\\\_test/accnet.onnx \\\\  --features 0.1 0.2 0.3 0.25 0.24 0.22 0.21 0.20 0.18 0.17 0.15```
The 11 input values correspond to:
`v\\\_t`
`v\\\_t-1`
...
`v\\\_t-10`
Output Directories
The results from the three solved versions can be found in these directories:
`results/acc\\\_test`
`results/acc\\\_v2\\\_test`
`results/acc\\\_v3\\\_test`
These folders contain output files such as:
accuracy plots
loss plots
ONNX models
summary files
Examples include:
`acc\\\_accuracy.png`
`acc\\\_loss.png`
`accnet.onnx`
`summary.txt`
`acc\\\_v2\\\_accuracy.png`
`acc\\\_v2\\\_loss.png`
`accnet\\\_v2.onnx`
`acc\\\_v3\\\_accuracy.png`
`acc\\\_v3\\\_loss.png`
`accnet\\\_v3.onnx`
Summary of Results
This part was solved in three different versions by changing epochs and other parameters in order to improve classification accuracy.
Observed results from the completed runs were approximately:
Version 1: about 87%
Version 2: about 78%
Version 3: about 83%
Therefore, the first version gave the strongest accuracy among the tested configurations, while the later versions were useful for exploring richer features, different train/test strategies, and regularization choices.
Notes
The different versions were intentionally kept to compare how feature design, history length, model complexity, and training parameters affect the final ACC classification performance.
This makes the work useful not only as a final classifier, but also as an experimental comparison of three solution strategies.