# How to use llm inference function

## Model Source
The model used in this example come from:  
https://github.com/airockchip/rknn3_model_zoo/examples/Qwen2_5

Run export_llm.py to generate the corresponding model and config files in the model/llm directory, then copy the files over.

## Script Usage
*Usage:*
```
python test.py
```
*Description:*
- The default target platform in script is 'rk1820', please modify the 'target_platform' parameter of 'rknn.config' according to the actual platform.


## Expected Results
This example will print the results of the llm.inference as follows:
```
--> response:
 相对论是爱因斯坦提出的一系列关于重力、时间膨胀和长度收缩等现象的理论。它包括狭义相对论（1905年）和广义相对论（1915年）。以下是相对论的一些基本概念：

1. **狭义相对论**：
   - **质能等价原理**：在任何惯性参考系中，质量与能量之间存在等价关系。
   - **光速不变原理**：在所有惯性参考系中，光速在真空中的速度都是常数，约为299,796,458米/秒。

2. **广义相对论**：
   - **引力场**：时空结构由物质的质量和能量组成，这种质量分布决定了引力场的存在。
   - **时空弯曲**：物体在空间中的运动可以影响周围时空的弯曲程度，从而产生引力效应。
   - **时空曲率**：时空的弯曲会导致光线路径的变化，即所谓的“引力透镜”效应。

3. **特殊相对论**：
   - **牛顿力学**：适用于低速或接近真空的情况下的物理定律。
   - **相对性原理**：在不同的惯性参考系中，物理定律是相同的。

4. **普朗克-玻尔兹模型**：
   - **量子化**：原子和分子的行为可以用量子力学来描述，其中粒子的能量和动量以波的形式表现出来。
   - **热力学第二定律**：熵增加，导致系统趋向于无序状态。

这些理论不仅改变了我们对宇宙的理解，而且也推动了现代物理学的发展。它们为现代科技提供了基础框架，并且至今仍在不断更新和完善。
```
- Note: Different platforms, different versions of tools and drivers may have slightly different results.