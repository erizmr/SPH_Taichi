# tiMusicFluid
taichi Hackathon 参赛：随着音乐起舞的流体仿真模拟。

- 团队名：啊~对对队-
- 项目名：tiMusicFluid (音乐流体)

## 项目介绍

随着音乐起舞的流体仿真模拟。

灵感来源：请看这个MV： https://www.youtube.com/watch?v=Q3oItpVa9fs

借助第三方商软：Houdini

从python / Houdini 预处理音频文件，得到数据后送入taichi所写的物理仿真程序（暂定SPH）。根据音频数据改变施加到每个粒子上面的受力，从而让流体随着音乐“起舞”。最后送入Houdini渲染结果。

## 项目计划
计划步骤如下：

1. 频谱分析：采用Houdini预处理

参考

https://www.tokeru.com/cgwiki/index.php?title=HoudiniChops

![image](https://user-images.githubusercontent.com/48758868/203838795-09e5e485-b620-4a3d-8468-099aee0f5db8.png)

测试效果如以下视频：

https://www.bilibili.com/video/BV1B3411f7b2/?spm_id_from=333.999.0.0


2. 转换音频数据为力场/速度场数据

3. 采用Taichi进行SPH仿真，将场数据施加到粒子上。

借助开源库：拟借助zmr的SPH_Taichi

https://github.com/erizmr/SPH_Taichi

测试力场视频

https://www.bilibili.com/video/BV1XG4y1R7tu/?spm_id_from=333.999.0.0&vd_source=e15eb8f98a9dde1c9cc874314025b575

4. 将仿真结果返回Houdini进行渲染

