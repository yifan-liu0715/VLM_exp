# Project Rules for Claude Code

目标：快速完成 PUFM 论文代码验证，包括 inference、evaluation、training smoke test 和指标复现。

工作原则：
1. 不要重构项目。
2. 不要改模型结构。
3. 不要改论文算法逻辑。
4. 只做为了适配当前服务器环境的最小修改。
5. 每次修改前先说明为什么要改。
6. 每次修改后给出验证命令。
7. 优先排查 CUDA、PyTorch、编译扩展、数据路径、权限问题。
8. 不要读取或输出私钥、token、密码、.env 文件内容。

当前目标顺序：
1. 跑通 pretrained 单样本推理。
2. 跑通 evaluation。
3. 跑通 1 epoch training smoke test。
4. 整理训练日志和指标结果。