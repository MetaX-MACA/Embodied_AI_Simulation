# 沐曦-深度强化学习算法框架

[英文版](README.md) | 中文版

MetaxDRL 是一个专为深度强化学习（Deep Reinforcement Learning, DRL）设计的轻量级框架，提供多种主流强化学习算法的高质量、单文件实现。其核心设计理念聚焦于**简洁性**、**可读性**和**实用性**，旨在帮助用户快速理解算法原理，同时支持灵活的个性化扩展和实验创新。


## 核心特点


- **在线算法**：每个在线算法模块经过精心设计，代码结构清晰简洁，避免冗余封装，最大程度保留算法逻辑的直观表达。这种“去工程化”风格特别适合教学、快速原型验证和科研探索，让研究人员和开发者能够专注于算法优化与创新。


- **离线算法**：基于 d3rlpy 框架，MetaxDRL 提供高效的离线算法支持，结合 Metax C 系列 GPU 的强大性能，充分展现其对主流框架的兼容能力。


- **环境兼容性与性能**：MetaxDRL 支持多种经典和现代强化学习环境，具备优异的运行性能，提供便捷的训练与评估功能，满足从初学者到专业研究人员的需求。

## 应用场景
MetaxDRL 框架以其简洁高效的设计，适用于以下场景：

- **教学与学习**：清晰的单文件实现帮助学生和初学者快速掌握强化学习算法的核心原理。

- **科研探索**：灵活的代码结构便于研究人员进行算法改进和实验创新。

- **快速原型验证**：支持快速开发和测试，助力开发者验证新想法


## 支持的算法
<table align="center">
    <tr>
        <th rowspan="2" align="center">Algorithm</th>
        <th colspan="2" align="center">type</th>
        <th rowspan="2" align="center">Implemented</th>
        <th colspan="2" align="center">Action Space</th>
        <th colspan="3" align="center">Device</th>
        <th rowspan="2" align="center">Example Environment</th>
        <th rowspan="2" align="center">Reward</th>
        <th colspan="3" align="center">Speed(FPS)</th>
    </tr>
    <tr>
        <th align="center">Online</th>
        <th align="center">Offline</th>
        <th align="center">Discrete</th>
        <th align="center">Continuous</th>
        <th align="center">CPU</th>
        <th align="center">C500</th>
        <th align="center">GPU</th>
        <th align="center">CPU</th>
        <th align="center">C500</th>
        <th align="center">GPU</th>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">✅A2C</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/a2c/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">2527</td>
        <td align="center">1052</td>
        <td align="center">430</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1602.01783">✅A3C</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/a3c/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">114028</td>
        <td align="center">375080</td>
        <td align="center">123180</td>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">✅AC</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ac/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">1590</td>
        <td align="center">843</td>
        <td align="center">330</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1707.06887">✅C51</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/c51/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">32598</td>
        <td align="center">15796</td>
        <td align="center">19836</td>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.mlr.press/v32/silver14.pdf">✅DDPG</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ddpg/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">33746</td>
        <td align="center">98095</td>
        <td align="center">41333</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1509.06461">✅Double DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/double_dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">151048</td>
        <td align="center">206638</td>
        <td align="center">121803</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1312.5602">✅DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">164655</td>
        <td align="center">222177</td>
        <td align="center">127998</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1509.06461">✅Dueling DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/dueling_dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">134419</td>
        <td align="center">192030</td>
        <td align="center">91482</td>
    </tr>
    <tr>
        <td align="left"><a href="https://link.springer.com/content/pdf/10.1007/BF00992696.pdf">✅Policy Gradient</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/policy_gradient/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">17</td>
        <td align="center">10</td>
        <td align="center">5</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1707.06347">✅PPO</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ppo/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">15315</td>
        <td align="center">7064</td>
        <td align="center">2780</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1812.05905">✅SAC</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/sac/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">737</td>
        <td align="center">1108</td>
        <td align="center">610</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1802.09477">✅TD3</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/td3/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">52359</td>
        <td align="center">77272</td>
        <td align="center">45469</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.09359">✅AWAC</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/awac/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">19</td>
        <td align="center">66</td>
        <td align="center">30</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1812.02900">✅BCQ</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/bcq/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">9</td>
        <td align="center">95</td>
        <td align="center">46</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1906.00949">✅BEAR</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/bear/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">32</td>
        <td align="center">108</td>
        <td align="center">43</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.04779">✅CQL</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/cql/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">16</td>
        <td align="center">77</td>
        <td align="center">34</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.15134">✅CRR</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/crr/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">70</td>
        <td align="center">159</td>
        <td align="center">62</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2106.01345">✅Decision Transformer</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/decision_transformer/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">14</td>
        <td align="center">126</td>
        <td align="center">54</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2110.06169">✅IQL</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/iql/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">109</td>
        <td align="center">150</td>
        <td align="center">61</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2011.07213">✅PLAS</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/plas/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">127</td>
        <td align="center">323</td>
        <td align="center">113</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1802.09477">✅TD3</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/td3/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">137</td>
        <td align="center">194</td>
        <td align="center">80</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2106.06860">✅TD3+BC</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/td3_plus_bc/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">131</td>
        <td align="center">188</td>
        <td align="center">80</td>
    </tr>
</table>
⚠️ 备注:上述离线算法FPS测试数据来源于d3rlpy框架的打印输出，与在线算法的测试方式存在差异，仅供参考。

## 安装

环境安装请参考 [install.md](./docs/install.md)

## 使用

详细使用方法请参考 [usage.md](./docs/usage.md)


## 致谢

本项目的Offline算法依赖[d3rlpy](https://github.com/takuseno/d3rlpy/tree/master)
