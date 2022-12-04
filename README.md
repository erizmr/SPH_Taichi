# tiMusicFluid
taichi Hackathon 参赛：随着音乐起舞的流体仿真模拟。

- 团队名：啊~对对队-
- 项目名：tiMusicFluid (音乐流体)

## 项目介绍

随着音乐起舞的流体仿真模拟。

从python / Houdini 预处理音频文件，得到数据后送入taichi所写的物理仿真程序（SPH）。根据音频数据改变施加到每个粒子上面的受力，从而让流体随着音乐“起舞”。最后送入Houdini渲染结果。

灵感来源：请看这个MV： https://www.youtube.com/watch?v=Q3oItpVa9fs


## 项目步骤

1. 对音乐进行频谱分析和采样

2. 转换音频数据为冲量，结合taichi的流体引擎进行仿真

3. 渲染仿真结果


## 借助软件/项目
商软（用于音频处理和渲染）：Houdini
开源库（流体引擎）：zmr的SPH_Taichi https://github.com/erizmr/SPH_Taichi

## 结果
https://user-images.githubusercontent.com/48758868/205440079-aa5fb6e1-3840-4419-b187-56ced810dd48.mp4
