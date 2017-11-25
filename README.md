# SegNet- tensorflow implementation
Training loss, training accuracy, validation loss, validation accuracy, test accuracy and MoI for different models.


|     Statistics     | SegNet Scartch| SegNet Vgg | SegNet Bayes Scratch | SegNet Bayes Vgg |
| ------------------ | ------------- | ---------- | -------------------- | ---------------- |
| Training Loss      |    0.02049    |  0.032013  |   0.05445            |                  |
| Training Accuracy  |    0.99189    |  0.987212  |   97.885%            |                  |
| Validation Loss    |    0.90312    |  0.897424  |   0.53518            |                  |
| Validation Accuracy|    0.87108    |  0.876878  |   90.331%            |                  |
| Test Accuracy      |    0.81328    |  0.817850  |   82.674%            |                  |
| Test MoI           |    0.42495    |  0.437668  |   0.4750             |                  |


Class average accuracy for different model:

| Method       | Sky   | Building| Column Pole| Road | Side-Walk | Tree | Sign Symbol| Fence | Car  | Pedestrain | Bicyclist |
| ------------ | ------| ------- | -----------| ---- | --------- | ---- | -----------| ----- | ---  | ---------- | --------- |
| +scratch     | 0.8946| 0.6859  | 0.0930     |0.8572| 0.5525    |0.5978| 0.1842     | 0.1153|0.6208| 0.1567     | 0.0965    |
| +vgg         | 0.8942| 0.6866  | 0.1102     |0.8728| 0.6266    |0.5910| 0.1653     | 0.1251|0.5796| 0.1882     | 0.1758    |
| +scratch(bay)| 90.05 | 69.13   | 15.75      |88.66 | 65.22     |59.79 | 25.85      | 14.68 |68.95 | 24.73      | 22.44     |

