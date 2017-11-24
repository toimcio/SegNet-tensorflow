# SegNet- tensorflow implementation
Training loss, training accuracy, validation loss, validation accuracy, test accuracy and MoI for different models.
|     Statistics     | SegNet Scartch| SegNet Vgg | SegNet Bayes Scratch | SegNet Bayes Vgg |
| ------------------ |--------------------------------------------------------------------- |
|                    |                   Without Class Balancing                            |
| ------------------ | ------------- | ---------- | -------------------- | ---------------- |
| Training Loss      |    0.02049    |  0.032013  | 
| Training Accuracy  |    0.99189    |  0.987212  |
| Validation Loss    |    0.90312    |  0.897424  |
| Validation Accuracy|    0.87108    |  0.876878  |
| Test Accuracy      |    0.81328    |  0.817850  |
| Test MoI           |    0.42495    |  0.437668  |

Class average accuracy for test images with SegNet-Vgg:
[0.89423, 0.68660, 0.11015, 0.87282, 0.62664, 0.59101, 0.16527, 0.12508, 0.57958, 0.18818, 0.17576, 0.23668]
Class average accuracy for test images with SegNet-scratch:
[0.89462, 0.68589, 0.09300, 0.85722, 0.55251, 0.59783, 0.18422, 0.11530, 0.62079, 0.15672, 0.09652, 0.24488]
