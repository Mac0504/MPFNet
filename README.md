# MPFNet: A Multi-Prior Fusion Network with a Progressive Training Strategy for Micro-Expression Recognition

# Abstract

Micro-expression recognition (MER), a critical subfield of affective computing, presents greater challenges than macro-expression recognition due to its brief duration and low intensity. While incorporating prior knowledge has been shown to enhance MER performance, existing methods predominantly rely on simplistic, singular sources of prior knowledge, failing to fully exploit multi-source information. This paper introduces the Multi-Prior Fusion Network (MPFNet), leveraging a progressive training strategy to optimize MER tasks. We propose two complementary encoders: the Generic Feature Encoder (GFE) and the Advanced Feature Encoder (AFE), both based on Inflated 3D ConvNets (I3D) with Coordinate Attention (CA) mechanisms, to improve the model’s ability to capture spatiotemporal and channel-specific features. Inspired by developmental psychology, we present two variants of MPFNet—MPFNet-P and MPFNet-C—corresponding to two fundamental modes of infant cognitive development: parallel and hierarchical processing. These variants enable the evaluation of different strategies for integrating prior knowledge. Extensive experiments demonstrate that MPFNet significantly improves MER accuracy while maintaining balanced performance across categories, achieving accuracies of 0.811, 0.924, and 0.857 on the SMIC, CASME II, and SAMM datasets, respectively. To the best of our knowledge, our approach achieves state-of-the-art performance on the SMIC and SAMM datasets. The source code is available at: https://github.com/Mac0504/MPFNet.

# Models
![image](pics/pic1.png)

# Results
![image](pics/pic2.png)

![image](pics/pic3.png)
