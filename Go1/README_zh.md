# 沐曦-具身智能复杂技能仿真解决方案

[英文版](README.md) | 中文版

## 介绍
本方案面向具身智能研究领域，依托沐曦C系列GPU强大的训练与推理能力，结合MuJoCo高精度仿真，通过将复杂技能拆解为简单且独立的动作单元，实现复杂技能在MuJoCo中的高效迁移与训练。为后续基于MuJoCo开展复杂技能训练，或从Isaac Gym迁移技能至MuJoCo环境，提供了可行且高效的最佳实践。

本方案通过四阶段流程: 资产对齐、功能接口替换、action动作调整、动作分解，完成在MuJoCo环境中的复杂技能迁移与训练。通过动作处理（如对称化/幅度调节），进一步加速技能学习过程，并有效提升训练的稳定性与鲁棒性。

我们以机械狗后空翻技能为例，来展示本方案的具体应用。


![quadruped](./imgs/quadruped.jpg)

受[Stage-Wise-CMORL](https://github.com/rllab-snu/Stage-Wise-CMORL)启发，本项目将仿真引擎从Issac Gyn迁移至MuJoCo，并遵循Stage-Wise-CMORL的流程进行任务训练。

效果演示：

![demo](./imgs/demo.gif)

## 安装

环境安装请参考 [install.md](./docs/install.md)

## 使用

详细使用方法请参考 [usage.md](./docs/usage.md)

## Mujoco迁移注意事项

详细注意事项请参考 [migration.md](./docs/migration.md)

## 致谢

本项目灵感来自 [Stage-Wise-CMORL](https://github.com/rllab-snu/Stage-Wise-CMORL)
